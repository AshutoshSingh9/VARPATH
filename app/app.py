import streamlit as st
import torch
import numpy as np
import pandas as pd
import sys
import os
import pickle
import yaml
import requests

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.feature_engineering import extract_features
from src.model import NIVEPNetwork
from src.uncertainty import predict_with_uncertainty

st.set_page_config(
    page_title="VARPATH",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- hide chrome ---- */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"]  { display: none; }
[data-testid="collapsedControl"] { display: none; }

/* ---- global font + background ---- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  font-family: 'Inter', sans-serif !important;
  background: #0a0d12 !important;
  color: #e4e8f0 !important;
}

/* ---- remove default block padding ---- */
.block-container {
  padding: 0 0 0 0 !important;
  max-width: 100% !important;
}

/* ---- canvas sits behind everything ---- */
#dot-canvas {
  position: fixed; top: 0; left: 0;
  width: 100vw; height: 100vh;
  z-index: 0; pointer-events: none;
  background: #0a0d12;
}

/* ---- all streamlit content above canvas ---- */
[data-testid="stAppViewContainer"] > * { position: relative; z-index: 1; }

/* ---- nav bar ---- */
.varpath-nav {
  display: flex; align-items: center;
  padding: 14px 40px; margin-bottom: 0;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  background: rgba(10,13,18,0.85);
  backdrop-filter: blur(10px);
}
.nav-logo { font-size: 0.95rem; font-weight: 700; letter-spacing: 0.14em; color: #fff; text-transform: uppercase; }
.nav-divider { width: 1px; height: 14px; background: rgba(255,255,255,0.12); margin: 0 14px; }
.nav-sub { font-size: 0.68rem; color: #4b5563; letter-spacing: 0.1em; text-transform: uppercase; }

/* ---- section label ---- */
.sect-label {
  font-size: 0.6rem; font-weight: 600; letter-spacing: 0.16em;
  color: #374151; text-transform: uppercase;
  margin: 20px 0 8px 0; padding-bottom: 6px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}

/* ---- left column inputs ---- */
[data-testid="column"]:first-child {
  border-right: 1px solid rgba(255,255,255,0.06);
  padding: 24px 24px !important;
  background: rgba(10,13,18,0.7);
  min-height: calc(100vh - 49px);
}
[data-testid="column"]:last-child {
  padding: 36px 48px !important;
  background: rgba(10,13,18,0.4);
  min-height: calc(100vh - 49px);
}

/* ---- inputs ---- */
.stTextInput input, .stNumberInput input {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 6px !important;
  color: #e4e8f0 !important;
  font-size: 0.8rem !important;
  font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: #1d4ed8 !important;
  box-shadow: 0 0 0 2px rgba(29,78,216,0.15) !important;
}
.stSelectbox > div > div {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 6px !important;
  color: #e4e8f0 !important;
  font-size: 0.8rem !important;
}
label, .stMarkdown p {
  font-size: 0.78rem !important;
  color: #6b7280 !important;
  margin-bottom: 2px !important;
}

/* ---- buttons ---- */
.stButton > button {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: #9ca3af !important;
  border-radius: 6px !important;
  font-size: 0.75rem !important;
  font-weight: 500 !important;
  font-family: 'Inter', sans-serif !important;
  letter-spacing: 0.04em !important;
  padding: 6px 14px !important;
  width: 100% !important;
  transition: all 0.15s ease !important;
  margin-top: 4px !important;
}
.stButton > button:hover {
  background: rgba(255,255,255,0.08) !important;
  border-color: rgba(255,255,255,0.24) !important;
  color: #fff !important;
}
[data-testid="baseButton-primary"] {
  background: #1d4ed8 !important;
  border-color: #1d4ed8 !important;
  color: #fff !important;
  font-weight: 600 !important;
  margin-top: 16px !important;
}
[data-testid="baseButton-primary"]:hover {
  background: #1e40af !important;
  border-color: #1e40af !important;
}

/* ---- slider ---- */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
.stSlider [role="slider"] { background: #1d4ed8 !important; }

/* ---- metrics ---- */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 8px !important;
  padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] { font-size: 0.62rem !important; color: #4b5563 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { font-size: 1.5rem !important; color: #e4e8f0 !important; font-weight: 700 !important; }

/* ---- spinner ---- */
.stSpinner { color: #1d4ed8 !important; }

/* ---- divider ---- */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 0 !important; }

/* ---- caption ---- */
.stCaption p { font-size: 0.68rem !important; color: #374151 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; margin-bottom: 2px !important; }

/* ---- hero type ---- */
.hero-title { font-size: 2.4rem; font-weight: 800; line-height: 1.1; letter-spacing: -0.02em; color: #fff; margin: 0 0 8px 0; }
.hero-sub { font-size: 0.8rem; color: #4b5563; letter-spacing: 0.04em; margin-bottom: 32px; }

/* ---- result cards ---- */
.rcard-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 24px; }
.rcard { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 18px 16px; }
.rcard-lbl { font-size: 0.58rem; color: #4b5563; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 8px; }
.rcard-val { font-size: 1.9rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
.rcard-sub { font-size: 0.65rem; color: #4b5563; }

/* ---- interp blocks ---- */
.iblock { border-radius: 8px; padding: 12px 16px; font-size: 0.78rem; line-height: 1.65; margin-top: 8px; }
.iblock-p { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.18); color: #fca5a5; }
.iblock-b { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.18); color: #6ee7b7; }
.iblock-n { background: rgba(59,130,246,0.06); border: 1px solid rgba(59,130,246,0.12); color: #7dd3fc; }

/* ---- AF / examples tables ---- */
.mini-table { width: 100%; border-collapse: collapse; font-size: 0.75rem; margin-top: 10px; }
.mini-table th { font-size: 0.6rem; color: #374151; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; padding: 6px 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.06); }
.mini-table td { padding: 7px 10px; border-bottom: 1px solid rgba(255,255,255,0.03); color: #9ca3af; vertical-align: top; }
.mini-table td:first-child { color: #d1d5db; font-family: monospace; }

/* ---- links ---- */
.ext-link { font-size: 0.73rem; color: #1d4ed8 !important; text-decoration: none; line-height: 2.2; display: block; }
.ext-link:hover { color: #3b82f6 !important; }
</style>

<!-- Particle dot canvas -->
<canvas id="dot-canvas"></canvas>
<script>
(function(){
  var c=document.getElementById('dot-canvas'),ctx=c.getContext('2d'),W,H,dots=[],mx=-999,my=-999,SP=30,PR=1.1,PL=100;
  function resize(){W=c.width=window.innerWidth;H=c.height=window.innerHeight;build();}
  function build(){dots=[];for(var y=SP/2;y<H;y+=SP)for(var x=SP/2;x<W;x+=SP)dots.push({ox:x,oy:y,x:x,y:y});}
  function frame(){
    ctx.clearRect(0,0,W,H);
    dots.forEach(function(d){
      var dx=mx-d.ox,dy=my-d.oy,dist=Math.sqrt(dx*dx+dy*dy),f;
      if(dist<PL){f=(PL-dist)/PL;f*=f;d.x=d.ox+dx*f*0.5;d.y=d.oy+dy*f*0.5;}
      else{d.x+=(d.ox-d.x)*0.1;d.y+=(d.oy-d.y)*0.1;}
      var b=dist<PL?0.12+0.3*(1-dist/PL):0.07;
      ctx.beginPath();ctx.arc(d.x,d.y,PR,0,Math.PI*2);
      ctx.fillStyle='rgba(59,130,246,'+b+')';ctx.fill();
    });
    requestAnimationFrame(frame);
  }
  window.addEventListener('mousemove',function(e){mx=e.clientX;my=e.clientY;});
  window.addEventListener('resize',resize);
  resize();frame();
})();
</script>
""", unsafe_allow_html=True)

# ─── NAV ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="varpath-nav">
  <span class="nav-logo">VARPATH</span>
  <span class="nav-divider"></span>
  <span class="nav-sub">Variant Pathogenicity Assessment Research Tool</span>
</div>
""", unsafe_allow_html=True)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
KNOWN_EFFECTS = ["stop_gained", "synonymous_variant", "missense_variant",
                 "frameshift_variant", "intron_variant", "splice_region_variant"]

def _parse_hit(hit):
    vtype = "missense_variant"; af = 0.0; cadd = 0.0
    ann = hit.get('snpeff', {}).get('ann', [])
    if isinstance(ann, dict): ann = [ann]
    if ann:
        eff = ann[0].get('effect', '')
        if eff in KNOWN_EFFECTS: vtype = eff
    gn = hit.get('gnomad_exome', {})
    if isinstance(gn, dict):
        ab = gn.get('af', {})
        if isinstance(ab, dict): af = float(ab.get('af', 0.0) or 0.0)
        elif isinstance(ab, (int, float)): af = float(ab)
    cb = hit.get('cadd', {})
    if isinstance(cb, dict): cadd = float(cb.get('phred', 0.0) or 0.0)
    return vtype, max(0.0, min(1.0, af)), max(0.0, min(50.0, cadd))

def _label(hit):
    rs = hit.get('dbsnp', {}).get('rsid', '')
    ann = hit.get('snpeff', {}).get('ann', [])
    if isinstance(ann, dict): ann = [ann]
    gene = ann[0].get('genename', '?') if ann else '?'
    eff  = ann[0].get('effect', '?') if ann else '?'
    return f"{gene} — {eff}" + (f"  [{rs}]" if rs else f"  [{hit.get('_id','')[:20]}]")

def _apply(hit):
    vt, af, cv = _parse_hit(hit)
    st.session_state.wvtype = vt
    st.session_state.waf    = af
    st.session_state.wcadd  = cv

@st.cache_resource
def load_assets():
    try:
        cfg = yaml.safe_load(open(os.path.join(base_dir, 'config.yaml')))
        md  = os.path.join(base_dir, 'models')
        enc = pickle.load(open(os.path.join(md, 'encoders.pkl'), 'rb'))
        dim = pickle.load(open(os.path.join(md, 'input_dim.pkl'), 'rb'))
        net = NIVEPNetwork(dim, cfg['model']['hidden_dims'], cfg['model']['dropout'])
        net.load_state_dict(torch.load(os.path.join(md, 'model.pt'), weights_only=True))
        net.eval()
        return cfg, net, enc
    except FileNotFoundError:
        return None, None, None

config, model, encoders = load_assets()

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for k, v in [('wvtype','synonymous_variant'),('waf',0.5),('wcadd',0.5),
             ('sr',[]),('sl',[])]:
    if k not in st.session_state: st.session_state[k] = v

# ─── CALLBACKS ────────────────────────────────────────────────────────────────
def cb_preset():
    p = st.session_state.preset
    if p == "Benign example":
        st.session_state.wvtype="synonymous_variant"; st.session_state.waf=0.85; st.session_state.wcadd=1.2
    elif p == "Pathogenic example":
        st.session_state.wvtype="stop_gained"; st.session_state.waf=0.0001; st.session_state.wcadd=35.0

def cb_gene():
    q = st.session_state.gi.strip()
    if not q: return
    try:
        r = requests.get(f"https://myvariant.info/v1/query?q={q}&fields=_id,dbsnp.rsid,snpeff.ann.effect,snpeff.ann.genename,cadd.phred,gnomad_exome.af.af&size=8", timeout=10).json()
        hits = r.get('hits',[])
        st.session_state.sr = hits
        st.session_state.sl = [_label(h) for h in hits]
    except: st.toast("API unavailable")

def cb_rsid():
    q = st.session_state.ri.strip()
    if not q: return
    try:
        r = requests.get(f"https://myvariant.info/v1/query?q={q}&fields=dbsnp.rsid,snpeff,cadd,gnomad_exome&size=1", timeout=10).json()
        hits = r.get('hits',[])
        if hits: _apply(hits[0])
        else: st.toast("Not found")
    except: st.toast("API unavailable")

def cb_clinvar():
    q = st.session_state.ci.strip()
    if not q: return
    try:
        r = requests.get(f"https://myvariant.info/v1/query?q=clinvar.rcv.accession:{q}&fields=dbsnp.rsid,snpeff,cadd,gnomad_exome&size=1", timeout=10).json()
        hits = r.get('hits',[])
        if hits: _apply(hits[0])
        else: st.toast("Not found")
    except: st.toast("API unavailable")

# ─── TWO-COLUMN LAYOUT ───────────────────────────────────────────────────────
left, right = st.columns([1.15, 2.85], gap="small")

# ══════════════════════════════════════════
# LEFT COLUMN — controls
# ══════════════════════════════════════════
with left:
    # Gene search
    st.markdown('<div class="sect-label">Gene / Keyword Search</div>', unsafe_allow_html=True)
    st.text_input("Gene or keyword", key="gi", placeholder="BRCA1, TP53, KRAS …", label_visibility="collapsed")
    st.button("Search", on_click=cb_gene, key="btn_gene")
    if st.session_state.sl:
        sel = st.selectbox("Results", st.session_state.sl, key="sel_gene", label_visibility="collapsed")
        idx = st.session_state.sl.index(sel)
        if st.button("Use this variant", key="btn_use"):
            _apply(st.session_state.sr[idx]); st.rerun()

    # rsID
    st.markdown('<div class="sect-label">Fetch by rsID</div>', unsafe_allow_html=True)
    st.text_input("rsID", key="ri", placeholder="rs121913527", label_visibility="collapsed")
    st.button("Fetch rsID", on_click=cb_rsid, key="btn_rsid")

    # ClinVar
    st.markdown('<div class="sect-label">Fetch by ClinVar RCV</div>', unsafe_allow_html=True)
    st.text_input("ClinVar RCV", key="ci", placeholder="RCV000013961", label_visibility="collapsed")
    st.button("Fetch ClinVar", on_click=cb_clinvar, key="btn_cv")

    # Presets
    st.markdown('<div class="sect-label">Presets</div>', unsafe_allow_html=True)
    st.selectbox("Preset", ["Custom", "Benign example", "Pathogenic example", "Fetched from Database"],
                 key="preset", on_change=cb_preset, label_visibility="collapsed")

    # Inputs
    st.markdown('<div class="sect-label">Variant Parameters</div>', unsafe_allow_html=True)

    st.caption("Variant Type")
    v_type = st.selectbox("Variant Type", KNOWN_EFFECTS, key="wvtype", label_visibility="collapsed",
        help="Functional consequence. stop_gained / frameshift = severe. synonymous = usually benign.")

    st.caption("Allele Frequency — gnomAD")
    st.session_state.waf = max(0.0, min(1.0, float(st.session_state.waf)))
    af = st.number_input("AF", min_value=0.0, max_value=1.0, step=0.001, format="%.4f",
                         key="waf", label_visibility="collapsed",
                         help=">5% = Benign (BA1). <0.1% = supports Pathogenic (PM2). Auto-filled from gnomAD.")

    st.caption("CADD Phred Score")
    st.session_state.wcadd = max(0.0, min(50.0, float(st.session_state.wcadd)))
    cadd = st.slider("CADD", min_value=0.0, max_value=50.0, step=0.1,
                     key="wcadd", label_visibility="collapsed",
                     help=">=20 = top 1% most deleterious. Auto-filled via API.")

    predict_btn = st.button("Run Analysis", type="primary", key="btn_run")

    # Links
    st.markdown('<div class="sect-label" style="margin-top:28px;">Data Sources</div>', unsafe_allow_html=True)
    st.markdown("""
    <a class="ext-link" href="https://gnomad.broadinstitute.org/" target="_blank">gnomAD — Allele Frequencies</a>
    <a class="ext-link" href="https://www.ncbi.nlm.nih.gov/clinvar/" target="_blank">ClinVar — Clinical Variants</a>
    <a class="ext-link" href="https://cadd.gs.washington.edu/snv" target="_blank">CADD — Deleteriousness Scores</a>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# RIGHT COLUMN — results / landing
# ══════════════════════════════════════════
with right:
    if not model:
        st.error("Model not found. Run `uv run python -m src.train` first.")

    elif predict_btn:
        with st.spinner("Running inference..."):
            X, _ = extract_features(
                pd.DataFrame([{'VariantType':v_type,'AlleleFrequency':af,'CADD_phred':cadd}]),
                fit=False, encoders=encoders)
            mp, vp = predict_with_uncertainty(model, torch.FloatTensor(X),
                                              num_samples=config['model']['mc_samples'])
        pv = float(np.squeeze(mp))
        sd = float(np.squeeze(np.sqrt(vp)))
        ip = pv > 0.5

        vc = "#ef4444" if ip else "#10b981"
        vt = "PATHOGENIC" if ip else "BENIGN"
        uc = "#f59e0b" if sd > 0.15 else "#10b981"

        st.markdown(f"""
        <div style="font-size:0.6rem;font-weight:600;letter-spacing:0.16em;color:#1d4ed8;text-transform:uppercase;margin-bottom:16px;">Analysis Results</div>
        <div class="rcard-grid">
          <div class="rcard" style="border-top:2px solid {vc};">
            <div class="rcard-lbl">Prediction</div>
            <div class="rcard-val" style="color:{vc};font-size:1.3rem;letter-spacing:0.06em;">{vt}</div>
            <div class="rcard-sub">threshold 0.5</div>
          </div>
          <div class="rcard">
            <div class="rcard-lbl">Pathogenicity Score</div>
            <div class="rcard-val">{pv:.3f}</div>
            <div class="rcard-sub">model output probability</div>
          </div>
          <div class="rcard" style="border-top:2px solid {uc};">
            <div class="rcard-lbl">Uncertainty sigma</div>
            <div class="rcard-val" style="color:{uc};">±{sd:.3f}</div>
            <div class="rcard-sub">{config['model']['mc_samples']} MC forward passes</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Interpretation
        if ip:
            reasons = []
            if cadd > 20: reasons.append("CADD above pathogenic threshold (>20)")
            if af < 0.01:  reasons.append("allele frequency supports PM2 (<1%)")
            if v_type in ["stop_gained","frameshift_variant"]: reasons.append("truncating variant")
            body = "Prediction: <strong>Pathogenic</strong>. Supporting evidence: " + (", ".join(reasons) if reasons else "complex feature interaction")
            cls = "iblock-p"
        else:
            reasons = []
            if af > 0.05: reasons.append("high population frequency (BA1)")
            if cadd < 15: reasons.append("low CADD — limited conservation impact")
            body = "Prediction: <strong>Benign</strong>. Supporting evidence: " + (", ".join(reasons) if reasons else "overall benign feature profile")
            cls = "iblock-b"

        st.markdown(f"""
        <div class="iblock {cls}" style="margin-top:16px;">{body}</div>
        <div class="iblock iblock-n">
          Uncertainty reflects Bayesian approximation via Monte Carlo Dropout.
          Values above 0.15 indicate low training-data density for this input — treat result with additional caution.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.6rem;font-weight:600;letter-spacing:0.16em;color:#1d4ed8;text-transform:uppercase;margin-bottom:8px;">Input Summary</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Variant Type", v_type)
        with c2: st.metric("Allele Frequency", f"{af:.4f}")
        with c3: st.metric("CADD Phred", f"{cadd:.1f}")

    else:
        # Landing
        st.markdown("""
        <div class="hero-title">Variant<br>Pathogenicity<br>Assessment</div>
        <div class="hero-sub">Statistical variant pathogenicity classification with uncertainty quantification</div>
        """, unsafe_allow_html=True)

        # AF guide
        st.markdown("""
        <div style="font-size:0.6rem;font-weight:600;letter-spacing:0.16em;color:#1d4ed8;text-transform:uppercase;margin-bottom:12px;">Allele Frequency Guide</div>
        <p style="font-size:0.76rem;color:#4b5563;line-height:1.7;margin-bottom:10px;">
          Allele Frequency is <strong style="color:#9ca3af;">automatically fetched from gnomAD</strong> when you use any lookup method.
          Novel variants absent from gnomAD default to 0.0 — a strong prerequisite for pathogenicity.
        </p>
        <table class="mini-table">
          <tr><th>AF Range</th><th>Interpretation</th><th>ACMG Evidence</th></tr>
          <tr><td>&gt; 5%</td><td>Very common in healthy populations</td><td>BA1 — Benign standalone</td></tr>
          <tr><td>1 – 5%</td><td>Low-frequency general variation</td><td>BS1 — Benign supporting</td></tr>
          <tr><td>0.1 – 1%</td><td>Rare; evidence inconclusive alone</td><td>VUS region</td></tr>
          <tr><td>0.01 – 0.1%</td><td>Very rare; investigate further</td><td>PM2 — Pathogenic moderate</td></tr>
          <tr><td>&lt; 0.01%</td><td>Extremely rare or novel</td><td>PM2 — strong prerequisite</td></tr>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Examples
        st.markdown("""
        <div style="font-size:0.6rem;font-weight:600;letter-spacing:0.16em;color:#1d4ed8;text-transform:uppercase;margin-bottom:12px;">Quick-Start Examples</div>
        <table class="mini-table">
          <tr><th>Variant</th><th>Method</th><th>Query</th><th>Expected</th></tr>
          <tr><td>BRAF V600E</td><td>rsID</td><td>rs121913527</td><td>Pathogenic</td></tr>
          <tr><td>Common benign SNP</td><td>rsID</td><td>rs58991260</td><td>Benign</td></tr>
          <tr><td>ClinVar pathogenic</td><td>ClinVar</td><td>RCV000013961</td><td>Pathogenic</td></tr>
          <tr><td>TP53 variants</td><td>Gene search</td><td>TP53</td><td>Multiple results</td></tr>
          <tr><td>Manual benign</td><td>Manual</td><td>synonymous, AF=0.10, CADD=2.5</td><td>Benign</td></tr>
          <tr><td>Manual pathogenic</td><td>Manual</td><td>stop_gained, AF=0.00, CADD=45.0</td><td>Pathogenic</td></tr>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.6rem;font-weight:600;letter-spacing:0.16em;color:#1d4ed8;text-transform:uppercase;margin-bottom:12px;">Model Architecture</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;font-size:0.75rem;color:#4b5563;line-height:2.2;">
          <div>
            <span style="color:#9ca3af;font-weight:600;display:block;margin-bottom:2px;">Inputs</span>
            Variant Type — One-Hot encoded<br>
            Allele Frequency — StandardScaler<br>
            CADD Phred — StandardScaler + median imputation
          </div>
          <div>
            <span style="color:#9ca3af;font-weight:600;display:block;margin-bottom:2px;">Inference</span>
            MLP: input → 64 → 32 → 1 (sigmoid)<br>
            ReLU activations + Dropout (0.3)<br>
            20 stochastic MC Dropout forward passes
          </div>
        </div>
        """, unsafe_allow_html=True)
