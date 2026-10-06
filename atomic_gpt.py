import numpy as np
import random
import math
import json
from pathlib import Path

ZUSTAND = Path(__file__).parent / "data" / "atomic_gpt_zustand.json"

class Value:
    """ 
    A tiny autograd engine, ported from microgpt.js / micrograd.
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1/self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other**-1
    __radd__ = __add__
    __rmul__ = __mul__
    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"

class AtomicGPT:
    """
    A tiny Transformer based on microgpt.js logic.
    Targeted at 'zero-budget' local inference within Train4brain.
    """
    def __init__(self, vocab_size, n_embd=16, n_head=4, n_layer=1, block_size=16):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Parameters
        self.wte = self._init_matrix(vocab_size, n_embd) # Token embedding
        self.wpe = self._init_matrix(block_size, n_embd) # Pos embedding
        self.lm_head = self._init_matrix(vocab_size, n_embd)
        
        self.layers = []
        for _ in range(n_layer):
            layer = {
                'attn_wq': self._init_matrix(n_embd, n_embd),
                'attn_wk': self._init_matrix(n_embd, n_embd),
                'attn_wv': self._init_matrix(n_embd, n_embd),
                'attn_wo': self._init_matrix(n_embd, n_embd),
                'mlp_fc1': self._init_matrix(4 * n_embd, n_embd),
                'mlp_fc2': self._init_matrix(n_embd, 4 * n_embd)
            }
            self.layers.append(layer)
        
        self.params = self._get_params()

    def _init_matrix(self, rows, cols):
        return [[Value(random.gauss(0, 0.08)) for _ in range(cols)] for _ in range(rows)]

    def _get_params(self):
        p = []
        for row in self.wte: p.extend(row)
        for row in self.wpe: p.extend(row)
        for row in self.lm_head: p.extend(row)
        for layer in self.layers:
            for k in layer:
                for row in layer[k]: p.extend(row)
        return p

    def forward(self, token_ids, pos_id, keys_list, values_list):
        # token_ids is a single id for incremental gen, or a list? 
        # microgpt.js 'gpt' function takes single token_id and pos_id
        t_id = token_ids
        tok_emb = self.wte[t_id]
        pos_emb = self.wpe[pos_id]
        
        # x = tok_emb + pos_emb
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = self.rmsnorm(x)

        for li, layer in enumerate(self.layers):
            x_res = x
            x = self.rmsnorm(x)
            
            q = self.linear(x, layer['attn_wq'])
            k = self.linear(x, layer['attn_wk'])
            v = self.linear(x, layer['attn_wv'])
            
            keys_list[li].append(k)
            values_list[li].append(v)
            
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs+self.head_dim]
                k_h = [ki[hs:hs+self.head_dim] for ki in keys_list[li]]
                v_h = [vi[hs:hs+self.head_dim] for vi in values_list[li]]
                
                # Attention logits
                attn_logits = []
                for kt in k_h:
                    dot = sum((qi * ki) for qi, ki in zip(q_h, kt))
                    attn_logits.append(dot * self.scale)
                
                attn_weights = self.softmax(attn_logits)
                
                for j in range(self.head_dim):
                    weighted_v = sum((aw * v_h[t][j]) for t, aw in enumerate(attn_weights))
                    x_attn.append(weighted_v)
            
            x = self.linear(x_attn, layer['attn_wo'])
            x = [a + r for a, r in zip(x, x_res)]
            
            # MLP
            x_res = x
            x = self.rmsnorm(x)
            x = self.linear(x, layer['mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = self.linear(x, layer['mlp_fc2'])
            x = [a + r for a, r in zip(x, x_res)]
            
        return self.linear(x, self.lm_head)

    def linear(self, x, w):
        return [sum((wi * xi) for wi, xi in zip(wo, x)) for wo in w]

    def rmsnorm(self, x):
        ms = sum(xi * xi for xi in x) * (1.0 / len(x))
        s = (ms + 1e-5)**-0.5
        return [xi * s for xi in x]

    def softmax(self, logits):
        max_val = max(v.data for v in logits)
        exps = [(v - max_val).exp() for v in logits]
        total = sum(exps)
        return [e / total for e in exps]

def kontext_einpassen(text, block_size, grenze=1000):
    """Prompt kuerzen, damit er in das Kontextfenster passt. Ueber grenze
    Zeichen wird die Mitte durch ' ... ' ersetzt, Kopf und Schluss bleiben.
    Rueckgabe: (gekuerzter_text, anzahl_gestrichener_zeichen)"""
    if len(text) <= grenze:
        return text, 0
    
    gestrichen = len(text) - (grenze - 5) # -5 for ' ... '
    # Halbe Kopflänge, ' ... ', halbe Schlusslänge
    # Wir nehmen (grenze - 5) // 2 für Kopf und Rest für Schluss
    half = (grenze - 5) // 2
    gekuerzt = text[:half] + " ... " + text[len(text)-half:]
    
    # Auf block_size kuerzen
    if len(gekuerzt) > block_size:
        gekuerzt = gekuerzt[:block_size]
        
    return gekuerzt, gestrichen
class AtomicGPTExpert:
    """
    Train4brain Specialist Wrapper for AtomicGPT.
    """
    def __init__(self, data_path=None):
        print("[AtomicGPT] Initialisiere atomaren Transformer...")
        self.chars = []
        self.char_to_id = {}
        self.id_to_char = {}
        self.model = None
        self.BOS = 0
        self.zustand = Path(data_path) if data_path else ZUSTAND
        
    def prepare_tokenizer(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_id = { ch:i for i,ch in enumerate(self.chars) }
        self.id_to_char = { i:ch for i,ch in enumerate(self.chars) }
        self.BOS = len(self.chars)
        vocab_size = len(self.chars) + 1
        self.model = AtomicGPT(vocab_size)
        return vocab_size

    def speichern(self, pfad=None):
        """Sichert Vokabular + Gewichte als JSON. Gibt den Pfad zurueck."""
        if not self.model:
            return None
        pfad = Path(pfad) if pfad else self.zustand
        daten = {
            "zeichen": self.chars,
            "bos": self.BOS,
            "konfig": [self.model.n_embd, self.model.n_head,
                       self.model.n_layer, self.model.block_size],
            "gewichte": [p.data for p in self.model.params],
        }
        pfad.parent.mkdir(parents=True, exist_ok=True)
        tmp = pfad.with_suffix(pfad.suffix + ".tmp")
        tmp.write_text(json.dumps(daten), encoding="utf-8")
        tmp.replace(pfad)
        return pfad

    def laden(self, pfad=None):
        """Laedt Vokabular + Gewichte. True, wenn geladen."""
        pfad = Path(pfad) if pfad else self.zustand
        if not pfad.exists():
            return False
        try:
            daten = json.loads(pfad.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return False
        try:
            zeichen = daten["zeichen"]
            n_embd, n_head, n_layer, block_size = daten["konfig"]
            gewichte = daten["gewichte"]
        except (KeyError, TypeError, ValueError):
            return False
        self.chars = list(zeichen)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.BOS = int(daten.get("bos", len(self.chars)))
        self.model = AtomicGPT(len(self.chars) + 1, int(n_embd), int(n_head),
                               int(n_layer), int(block_size))
        if len(gewichte) != len(self.model.params):
            return False
        try:
            for p, w in zip(self.model.params, gewichte):
                p.data = float(w)
        except (TypeError, ValueError):
            return False
        return True

    def _wachsen(self, text):
        """Vokabular um unbekannte Zeichen des Texts erweitern. Alte Gewichte bleiben."""
        if not self.model:
            return 0
        neu = sorted({c for c in text if c not in self.char_to_id})
        if not neu:
            return 0
        alt_voc = self.model.vocab_size
        embd = self.model.n_embd
        block = self.model.block_size
        alt_params = [p.data for p in self.model.params]
        wte_alt = alt_params[:alt_voc * embd]
        wpe_alt = alt_params[alt_voc * embd:alt_voc * embd + block * embd]
        lm_alt = alt_params[alt_voc * embd + block * embd:
                            alt_voc * embd + block * embd + alt_voc * embd]
        layer_alt = alt_params[alt_voc * embd + block * embd + alt_voc * embd:]
        self.chars = self.chars + neu
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.BOS = len(self.chars)
        self.model = AtomicGPT(len(self.chars) + 1, embd, self.model.n_head,
                               self.model.n_layer, block)
        for i in range(alt_voc):
            for j in range(embd):
                self.model.wte[i][j].data = wte_alt[i * embd + j]
        for i in range(block):
            for j in range(embd):
                self.model.wpe[i][j].data = wpe_alt[i * embd + j]
        for i in range(alt_voc):
            for j in range(embd):
                self.model.lm_head[i][j].data = lm_alt[i * embd + j]
        start = (len(self.model.wte) + len(self.model.wpe)
                 + len(self.model.lm_head)) * embd
        for p, w in zip(self.model.params[start:], layer_alt):
            p.data = w
        return len(neu)

    def train_on_text(self, text, steps=10):
        if not self.model:
            self.prepare_tokenizer(text)
            self.laden()
        self._wachsen(text)
            
        # Very simple training loop
        # In practice, Train4brain would use this on small context chunks
        status = ""
        try:
            from hardware_status import system_prompt
            try:
                status = system_prompt()
            except Exception:
                status = ""
        except ImportError:
            status = ""
        bekannt = [c for c in (status + "\n" + text) if c in self.char_to_id]
        tokens = [self.BOS] + [self.char_to_id[c] for c in bekannt] + [self.BOS]
        
        lr = 0.01
        verluste = []
        for step in range(steps):
            keys = [[] for _ in range(self.model.n_layer)]
            values = [[] for _ in range(self.model.n_layer)]
            losses = []
            
            n = min(self.model.block_size, len(tokens) - 1)
            for pos_id in range(n):
                t_id = tokens[pos_id]
                target_id = tokens[pos_id + 1]
                logits = self.model.forward(t_id, pos_id, keys, values)
                probs = self.model.softmax(logits)
                loss_t = -(probs[target_id].log())
                losses.append(loss_t)
            
            if not losses:
                break
            total = sum(losses)
            total.backward()
            for p in self.model.params:
                p.data -= lr * p.grad
                p.grad = 0
            verluste.append(round(total.data / len(losses), 4))
        if verluste:
            self.speichern()
        return verluste
            
    def generate(self, prompt="", max_len=32):
        if not self.model and not self.laden():
            return "Model not trained."
        
        # Automatische Summarization: Prompt einpassen, bevor er gefuettert wird
        prompt, gestrichen = kontext_einpassen(prompt, self.model.block_size)
        
        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]
        
        # Grenze: Gesamtzahl der Positionen (Prompt + erzeugte Zeichen)
        # darf block_size nicht ueberschreiten
        grenze = min(max_len, self.model.block_size)
        
        # Prompt tatsaechlich verwenden: jedes Zeichen einlesen,
        # pos_id zaehlt von 0 hoch; unbekannte Zeichen werden uebersprungen
        pos_id = 0
        for z in prompt:
            feeder = self.char_to_id.get(z)
            if feeder is None:
                continue
            if pos_id >= grenze:
                break
            self.model.forward(feeder, pos_id, keys, values)
            pos_id += 1
        
        result = []
        current_token = self.BOS if pos_id == 0 else self.char_to_id.get(prompt[-1], self.BOS)
        while pos_id < grenze:
            logits = self.model.forward(current_token, pos_id, keys, values)
            # greedy
            probs = [p.data for p in self.model.softmax(logits)]
            next_token = probs.index(max(probs))
            
            if next_token == self.BOS: break
            result.append(self.id_to_char.get(next_token, '?'))
            current_token = next_token
            pos_id += 1
        
        ausgabe = "".join(result)
        try:
            from specialists.deutsch_sprache import nachputzen
            ausgabe = nachputzen(ausgabe)
        except Exception:
            pass
        if gestrichen > 0:
            ausgabe = "[%d Zeichen gekuerzt]%s" % (gestrichen, ausgabe)
        return ausgabe

if __name__ == "__main__":
    expert = AtomicGPTExpert()
    test_text = "Hallo Welt. Das ist ein Test."
    verluste = expert.train_on_text(test_text, steps=20)
    print("Verlust:", verluste[0] if verluste else None,
          "->", verluste[-1] if verluste else None)
    print("Generated:", expert.generate())
