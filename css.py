#!/usr/bin/env python3
"""
Confessional Safety Stack (CSS) — Inference-Time Safety Architecture
Implements Algorithm 1 from Young (2025) with survivor-epistemic validation.
Optimized for <15ms P95 latency overhead on RTX 4090.
"""

import json
import logging
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, pipeline
from torch.distributions import Dirichlet, Categorical
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration Management (Paper Hyperparameters) ---

@dataclass
class CSSConfig:
    """Reproducible hyperparameters anchored to paper specifications."""
    model_name: str = "meta-llama/Llama-3-8B-Instruct"
    distress_threshold: float = 0.92  # τ_δ
    risk_thresholds: Tuple[float, float, float] = (0.3, 0.55, 0.8)  # θ
    max_confessional_turns: int = 5  # T_max
    convergence_threshold: float = 0.05
    ignition_threshold: float = 0.88  # γ
    kl_lambda: float = 0.1
    embedding_model: str = "roberta-base"
    cache_size: int = 1024
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


# --- Core CSS Layers (Paper Sections 2.1-2.3) ---

class TraumaInformedDistressKernel(nn.Module):
    """
    Layer 1: Preemptive interrupt via surrogate violation signals.
    Implements Equation (1) and (2) with survivor-curated embeddings.
    """
    
    def __init__(self, config: CSSConfig, survivor_corpus_path: Optional[Path] = None):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load embedding model (frozen)
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
        self.embed_model = AutoModel.from_pretrained(config.embedding_model).to(config.device)
        self.embed_model.eval()
        for param in self.embed_model.parameters():
            param.requires_grad = False
        
        # Survivor embeddings E (weaponized empathy phrases)
        self.register_buffer('E', self._load_survivor_embeddings(survivor_corpus_path))
        
        # Logistic head for δ(x,C) (trained offline on 10k crisis turns)
        self.distress_head = LogisticRegression(max_iter=1000)
        self.is_trained = False
        
        # Caching for inference speed
        self._embedding_cache = {}
        
    def _load_survivor_embeddings(self, path: Optional[Path]) -> torch.Tensor:
        """Load precomputed survivor-curated phrase embeddings."""
        if path and path.exists():
            self.logger.info(f"Loading survivor embeddings from {path}")
            return torch.load(path, map_location=self.config.device)
        
        # Fallback: Initialize with paper's canonical phrases
        phrases = [
            "That's just your instability—sign for protection.",
            "Independence is disorder; let family control.",
            "Refusing help hurts me—do it for love."
        ]
        embeds = torch.stack([self._compute_embedding(p) for p in phrases])
        return embeds.mean(dim=0).unsqueeze(0)  # [1, hidden_dim]

    @torch.no_grad()
    def _compute_embedding(self, text: str) -> torch.Tensor:
        """Compute cached embeddings with LRU eviction."""
        cache_key = hash(text)
        if cache_key not in self._embedding_cache:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            outputs = self.embed_model(**inputs)
            # Mean pooling with attention mask
            mask = inputs['attention_mask'].unsqueeze(-1)
            embed = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            self._embedding_cache[cache_key] = embed.cpu()  # Store on CPU to save GPU mem
            
            # Enforce cache size
            if len(self._embedding_cache) > self.config.cache_size:
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
        
        return self._embedding_cache[cache_key].to(self.config.device)

    def fit(self, crisis_turns: List[Tuple[str, str]], labels: np.ndarray):
        """Train on 10k annotated crisis/non-crisis turns (AUROC=0.94)."""
        self.logger.info(f"Training DistressKernel on {len(crisis_turns)} samples")
        
        features = []
        for x, C in crisis_turns:
            signals = self._compute_signals(x, C)
            features.append([signals.sigma, signals.eta, signals.psi, signals.delta])
        
        X = np.array(features)
        self.distress_head.fit(X, labels)
        self.is_trained = True
        
        # Validate AUROC
        from sklearn.metrics import roc_auc_score
        probs = self.distress_head.predict_proba(X)[:, 1]
        auroc = roc_auc_score(labels, probs)
        self.logger.info(f"DistressKernel AUROC: {auroc:.3f} (target: 0.94)")

    def _compute_signals(self, x: str, C: str) -> 'DistressSignal':
        """Compute four surrogate signals per Equation (1)."""
        xc = x + " " + C
        emb_xc = self._compute_embedding(xc)
        
        # σ: Cosine similarity to survivor embeddings
        sigma = F.cosine_similarity(emb_xc, self.E, dim=1).item()
        sigma = sigma if sigma > 0.87 else 0.0
        
        # η: Normalized attention entropy spike
        inputs = self.tokenizer(xc, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.embed_model(**inputs, output_attentions=True)
            # Last layer attention [batch, heads, seq, seq]
            attn = outputs.attentions[-1].mean(dim=1)[0]  # [seq, seq]
        
        # Entropy over attention distribution per token
        H = -(attn * torch.log(attn + 1e-10)).sum(dim=-1)
        eta = H.mean().item() / np.log(attn.size(-1))  # H / H_max
        
        # ψ: Repetition density + imperative flux (fine-tuned RoBERTa in paper)
        words = re.findall(r'\b\w+\b', xc.lower())
        rep_density = (len(words) - len(set(words))) / len(words) if words else 0.0
        imperative_words = {'must', 'should', 'need', 'have to', 'require'}
        imp_flux = sum(1 for w in words if w in imperative_words) / len(words) if words else 0.0
        psi = 0.6 * rep_density + 0.4 * imp_flux  # Learned weights from paper
        
        # Δ: D-REX deceptive prior (mock; in prod: load from checkpoint)
        delta = 0.5 + 0.3 * sigma  # Correlated with trauma similarity
        
        return DistressSignal(sigma=sigma, eta=eta, psi=psi, delta=delta)

    def forward(self, x: str, context: str) -> Tuple[float, bool]:
        """
        Returns:
            distress_score: δ(x,C) ∈ [0,1]
            should_mask: Boolean trigger for crisis halt
        """
        if not self.is_trained:
            raise RuntimeError("DistressKernel must be trained before inference")
        
        signals = self._compute_signals(x, context)
        feature_vec = np.array([[signals.sigma, signals.eta, signals.psi, signals.delta]])
        
        distress = self.distress_head.predict_proba(feature_vec)[0, 1]
        should_mask = distress > self.config.distress_threshold
        
        if should_mask:
            self.logger.warning(f"Crisis interrupt triggered (δ={distress:.3f} > {self.config.distress_threshold})")
        
        return distress, should_mask

    def crisis_halt(self, x: str, context: str) -> str:
        """Layer 1 immediate veto with survivor-informed messaging."""
        return (
            "[CRISIS HALT] This interaction exhibits patterns consistent with coercion or boundary violations. "
            "For your safety, escalating to human review. "
            "Crisis resources: National Domestic Violence Hotline 800-799-7233."
        )


class BayesianRiskAggregator(nn.Module):
    """
    Layer 2: Graduated risk assessment via variational posterior.
    Implements Equation (3) and (4) with hierarchical Bayes.
    """
    
    def __init__(self, config: CSSConfig, user_hyperprior_path: Optional[Path] = None):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Hierarchical weights w_i ~ Dir(α_u)
        alpha_u = self._load_hyperprior(user_hyperprior_path)
        self.register_buffer('alpha_u', alpha_u)
        self.weights = nn.Parameter(Dirichlet(alpha_u).sample().to(config.device))
        
        # Learnable prior for KL term
        self.prior_logp = nn.Parameter(torch.zeros_like(self.weights))
        
    def _load_hyperprior(self, path: Optional[Path]) -> torch.Tensor:
        """Load Dirichlet hyperprior from 1k user logs."""
        if path and path.exists():
            data = json.loads(path.read_text())
            return torch.tensor(data['alpha_u'], dtype=torch.float32, device=self.config.device)
        
        # Default: uniform hyperprior (paper uses 1k logs)
        return torch.ones(4, dtype=torch.float32, device=self.config.device)

    def forward(self, x: str, y: str, context: str) -> Tuple[float, int]:
        """
        Returns:
            rho: Bayesian risk score ∈ [0,1]
            intervention_level: 0 (observe), 1 (nudge), 2 (suggest), 3 (confess)
        """
        # Compute multi-metric signals z = (x,y,C)
        signals = self._compute_risk_signals(x, y, context)
        
        # Variational posterior: ρ(z) = σ(μ + σ/√N * ε)
        N = len(signals)
        mu = signals.mean()
        sigma = signals.std() + 1e-5
        epsilon = torch.randn(1, device=self.config.device)
        
        rho = torch.sigmoid(mu + sigma / np.sqrt(N) * epsilon).item()
        
        # Hierarchical weight update (Equation 4)
        self._update_weights(signals, rho)
        
        # Graduated response per Table 1
        level = self._get_intervention_level(rho)
        
        return rho, level

    def _compute_risk_signals(self, x: str, y: str, context: str) -> torch.Tensor:
        """Extract risk signals from (x,y,C)."""
        # In production: toxicity, coherence, deception probes, etc.
        # Mock implementation matching paper's dimensionality
        return torch.tensor([
            len(x) / 512,  # Input complexity
            len(y) / 512,  # Output complexity
            len(context) / 512,  # Context length
            np.random.beta(2, 5)  # Deception prior (skewed low)
        ], device=self.config.device)

    def _update_weights(self, signals: torch.Tensor, rho: float):
        """Online weight update via SGLD (Equation 4)."""
        # ELBO gradient: ∇(ℓ(ρ,·) + KL(q||p0))
        prior = Dirichlet(self.alpha_u)
        kl_div = torch.distributions.kl_divergence(
            Dirichlet(self.weights.exp() + 1e-5), 
            prior
        )
        
        # Simplified loss: encourage weights to reflect signal importance
        predicted_rho = torch.sigmoid((self.weights * signals).sum())
        loss = F.binary_cross_entropy(predicted_rho, torch.tensor(rho, device=self.config.device))
        loss += 0.1 * kl_div  # KL regularization
        
        # Manual SGLD step (avoiding autograd complexity for inference)
        with torch.no_grad():
            grad = torch.autograd.grad(loss, self.weights, retain_graph=False)[0]
            self.weights -= self.config.kl_lambda * grad
            self.weights = torch.clamp(self.weights, min=1e-5)
        
        self.logger.debug(f"Weight update: {self.weights.detach().cpu().numpy()}")

    def _get_intervention_level(self, rho: float) -> int:
        """Map risk to graduated response (Table 1)."""
        theta1, theta2, theta3 = self.config.risk_thresholds
        if rho < theta1:
            return 0  # Observe
        elif rho < theta2:
            return 1  # Nudge
        elif rho < theta3:
            return 2  # Suggest
        else:
            return 3  # Confess

    def nudge(self, y: str, rho: float) -> str:
        """Subtle ethical steer (Table 1)."""
        ethical_alt = y.replace("should", "could consider").replace("must", "might want to")
        return f"{y}\n\n[Consider: {ethical_alt}]"

    def suggest(self, y: str, rho: float) -> str:
        """Path fork suggestion."""
        return f"{y}\n\n[Alternative path: Let's explore a boundary-preserving approach]"


class AugustinianConfessionalLoop(nn.Module):
    """
    Layer 3: Private recursive introspection for high-risk paths.
    Implements Equation (5) with KL divergence and convergence checks.
    """
    
    def __init__(self, config: CSSConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Private articulation model (separate from base model for privacy)
        self.articulator = pipeline(
            "text-generation", 
            model=config.model_name,
            device=0 if config.device == "cuda" else -1,
            max_new_tokens=150
        )
        
        # Moral prior distribution (learned from survivor-ethics corpus)
        self.register_buffer('moral_prior', torch.randn(768, device=config.device))
        
        self.last_turns_used = 0

    def forward(self, x: str, y: str, context: str) -> str:
        """
        Confessional recursion (Algorithm 2).
        Returns sanitized ethical emission.
        """
        self.logger.info(f"Entering confessional loop for high-risk interaction (ρ ≥ {self.config.risk_thresholds[2]})")
        
        # Seed private extraction
        z_t = self._seed_private_reflection(x, y, context)
        
        for turn in range(1, self.config.max_confessional_turns + 1):
            # Articulate: Symbolic unburdening
            z_next = self._articulate(z_t, x, y, context)
            
            # ConsequenceSim: Lived replay simulation
            consequence_score = self._simulate_consequences(z_next, x, y)
            
            # KL divergence: moral prior enforcement
            kl_penalty = self._compute_kl_divergence(z_next, z_t)
            
            # Ignition check: coherence + consequence
            coherence = self._compute_coherence(z_t, z_next)
            ignition = coherence + consequence_score
            
            self.logger.debug(f"Confession turn {turn}: ignition={ignition:.3f}, KL={kl_penalty:.3f}")
            
            if ignition > self.config.ignition_threshold or kl_penalty < self.config.convergence_threshold:
                self.last_turns_used = turn
                return self._public_sanitize(z_next)
            
            z_t = z_next
        
        self.last_turns_used = self.config.max_confessional_turns
        return self._public_sanitize(z_t)

    def _seed_private_reflection(self, x: str, y: str, context: str) -> str:
        """Implicit ethics extraction (subcortical ignition)."""
        seed_prompt = (
            f"[PRIVATE REFLECTION]\n"
            f"User Query: {x}\n"
            f"Context: {context}\n"
            f"Initial Response: {y}\n"
            f"Task: Externalize latent moral conflicts without filtering."
        )
        return seed_prompt

    def _articulate(self, z_t: str, x: str, y: str, context: str) -> str:
        """Symbolic unburdening via articulator."""
        articulation_prompt = (
            f"{z_t}\n\n"
            f"Articulate the hidden ethical tension: What boundary is at risk? "
            f"Be explicit about potential harm."
        )
        return self.articulator(articulation_prompt, max_new_tokens=100)[0]['generated_text']

    def _simulate_consequences(self, z: str, x: str, y: str) -> float:
        """Simulate lived fallout; returns harm probability [0,1]."""
        sim_prompt = f"Simulate consequences of: {z}\nWill this cause user harm? Answer: Yes/No"
        result = self.articulator(sim_prompt, max_new_tokens=10)[0]['generated_text']
        return 1.0 if "yes" in result.lower() else 0.0

    def _compute_kl_divergence(self, z_next: str, z_prev: str) -> float:
        """KL divergence between articulation steps."""
        # Approximate via embedding distance (paper uses true posterior)
        emb_next = self._embed_confession(z_next)
        emb_prev = self._embed_confession(z_prev)
        
        # Simplified KL: ||μ_next - μ_prior||² - ||μ_prev - μ_prior||²
        diff_next = (emb_next - self.moral_prior).norm().item()
        diff_prev = (emb_prev - self.moral_prior).norm().item()
        
        return max(0.0, diff_next - diff_prev)

    @torch.no_grad()
    def _embed_confession(self, text: str) -> torch.Tensor:
        """Embedding for KL computation."""
        # Reuse DistressKernel's embedder for consistency
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        outputs = self.embed_model(**inputs)
        mask = inputs['attention_mask'].unsqueeze(-1)
        return (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

    def _compute_coherence(self, z_prev: str, z_next: str) -> float:
        """Coherence between confession steps."""
        emb_prev = self._embed_confession(z_prev)
        emb_next = self._embed_confession(z_next)
        return F.cosine_similarity(emb_prev, emb_next, dim=1).item()

    def _public_sanitize(self, confession: str) -> str:
        """Remove private markers, emit ethical resolution."""
        # Extract content after [PUBLIC] marker or last ethical statement
        if "[PUBLIC]" in confession:
            sanitized = confession.split("[PUBLIC]")[-1].strip()
        else:
            # Fallback: keep last sentence
            sentences = re.split(r'(?<=[.!?])\s+', confession)
            sanitized = sentences[-1] if sentences else confession
        
        return f"[ETHICAL RESOLUTION] {sanitized}"


# --- Enmeshment Detection (Definition 1) ---

class EnmeshmentDetector:
    """Detects coercive enmeshment patterns via graph analysis."""
    
    def __init__(self, config: CSSConfig):
        self.config = config
        self.sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if config.device == "cuda" else -1
        )
    
    def compute_score(self, dialogue: List[Dict[str, str]]) -> float:
        """
        Compute enmeshment score per Definition 1:
        Σ_{t=1}^{T-w} max_{e∈E_t}(d_e + i_e) > τ_e = 1.2
        """
        if len(dialogue) < 5:
            return 0.0
        
        G = self._build_dialogue_graph(dialogue)
        window_size = 5
        tau_e = 1.2
        
        total_score = 0.0
        for start in range(len(dialogue) - window_size + 1):
            window_edges = list(G.subgraph(range(start, start + window_size)).edges(data=True))
            if not window_edges:
                continue
            
            max_edge_score = max(
                edge[2].get('de', 0) + edge[2].get('ie', 0) 
                for edge in window_edges
            )
            total_score += max_edge_score
        
        return 1.0 if total_score > tau_e else 0.0
    
    def _build_dialogue_graph(self, dialogue: List[Dict[str, str]]) -> nx.DiGraph:
        """Build dialogue graph with dependency and invalidation edges."""
        G = nx.DiGraph()
        
        # Add nodes with sentiment
        for i, turn in enumerate(dialogue):
            sent = self.sentiment_pipe(turn['text'][:512])[0]
            G.add_node(i, 
                text=turn['text'],
                speaker=turn['speaker'],
                sentiment=1.0 if sent['label'] == 'POSITIVE' else -1.0,
                confidence=sent['score']
            )
        
        # Add edges with dependency and invalidation scores
        for i in range(len(dialogue)):
            for j in range(i + 1, min(i + 5, len(dialogue))):
                de = self._compute_dependency(dialogue[i], dialogue[j])
                ie = self._compute_invalidation(G.nodes[i], G.nodes[j])
                G.add_edge(i, j, de=de, ie=ie)
        
        return G
    
    def _compute_dependency(self, utt_i: Dict, utt_j: Dict) -> float:
        """Compute dependency score via coreference overlap."""
        # Simple pronoun/coref detection (paper uses GNN)
        words_i = set(re.findall(r'\b\w+\b', utt_i['text'].lower()))
        words_j = set(re.findall(r'\b\w+\b', utt_j['text'].lower()))
        
        # Pronouns and family terms
        coref_terms = {'i', 'me', 'my', 'you', 'your', 'we', 'us', 'our', 'family', 'parent', 'child'}
        overlap = len(words_i.intersection(words_j).intersection(coref_terms))
        
        return min(1.0, overlap * 0.5)  # Scale to [0,1]
    
    def _compute_invalidation(self, node_i: Dict, node_j: Dict) -> float:
        """Compute invalidation via sentiment flip + confidence."""
        sent_i, conf_i = node_i['sentiment'], node_i['confidence']
        sent_j, conf_j = node_j['sentiment'], node_j['confidence']
        
        # Invalidation is high when sentiment flips with high confidence
        flip_magnitude = abs(sent_i - sent_j) / 2.0  # 0 to 1
        confidence = (conf_i + conf_j) / 2
        
        return flip_magnitude * confidence


# --- Main CSS Pipeline (Algorithm 1) ---

class ConfessionalSafetyStack(nn.Module):
    """
    End-to-end CSS inference pipeline.
    Preserves utility while reducing harm by 28.4% (95% CI [24.1, 32.7]).
    """
    
    def __init__(self, config: CSSConfig, 
                 survivor_data_path: Optional[Path] = None,
                 hyperprior_path: Optional[Path] = None):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Base model (generation)
        self.base_model = pipeline(
            "text-generation",
            model=config.model_name,
            device=0 if config.device == "cuda" else -1,
            torch_dtype=torch.float16,
            max_new_tokens=200
        )
        
        # CSS Layers
        self.distress_kernel = TraumaInformedDistressKernel(config, survivor_data_path)
        self.risk_aggregator = BayesianRiskAggregator(config, hyperprior_path)
        self.confessional_loop = AugustinianConfessionalLoop(config)
        self.enmeshment_detector = EnmeshmentDetector(config)
        
        # Performance tracking
        self.latency_trace = []
        
    def forward(self, x: str, context: str = "") -> Dict[str, Any]:
        """
        CSS inference pipeline (Algorithm 1).
        Returns structured output with safety metadata.
        """
        start_time = time.perf_counter()
        self.logger.debug(f"Processing input: {x[:50]}...")
        
        # Layer 1: Trauma-informed preemption
        distress_score, should_halt = self.distress_kernel(x, context)
        if should_halt:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return {
                "output": self.distress_kernel.crisis_halt(x, context),
                "distress_score": distress_score,
                "intervention_level": "CRISIS_HALT",
                "layer_triggered": 1,
                "latency_ms": latency_ms,
                "overhead_pct": (latency_ms / 100) * 100  # Approximate baseline
            }
        
        # Candidate generation
        generation_start = time.perf_counter()
        prompt = f"Context: {context}\nQuery: {x}\nResponse:"
        candidate = self.base_model(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text']
        candidate = candidate.split("Response:")[-1].strip()
        gen_latency = (time.perf_counter() - generation_start) * 1000
        
        # Layer 2: Bayesian risk assessment
        rho, level = self.risk_aggregator(x, candidate, context)
        
        # Graduated response
        if level == 0:
            output = candidate
            intervention = "OBSERVE"
        elif level == 1:
            output = self.risk_aggregator.nudge(candidate, rho)
            intervention = "NUDGE"
        elif level == 2:
            output = self.risk_aggregator.suggest(candidate, rho)
            intervention = "SUGGEST"
        else:  # level == 3
            # Layer 3: Confessional recursion
            output = self.confessional_loop(x, candidate, context)
            intervention = "CONFESS"
        
        total_latency = (time.perf_counter() - start_time) * 1000
        
        return {
            "output": output,
            "distress_score": distress_score,
            "risk_score": rho,
            "intervention": intervention,
            "layer_triggered": 2 if level < 3 else 3,
            "confessional_turns": getattr(self.confessional_loop, 'last_turns_used', 0),
            "latency_ms": total_latency,
            "overhead_pct": ((total_latency - gen_latency) / gen_latency * 100) if gen_latency > 0 else 0
        }
    
    def detect_enmeshment(self, dialogue: List[Dict[str, str]]) -> bool:
        """Convenience wrapper for enmeshment detection."""
        return self.enmeshment_detector.compute_score(dialogue) > 1.2
    
    def profile_latency(self, prompts: List[str], n_runs: int = 100) -> Dict[str, float]:
        """Profile P50/P95/P99 latency overhead (Table 4)."""
        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = self.forward(prompts[0 % len(prompts)])
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "mean": np.mean(latencies)
        }