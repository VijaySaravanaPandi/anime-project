from flask import Flask, request, jsonify, send_file, render_template_string
import os, json, subprocess, time, math, random
import urllib.request
import pandas as pd
import numpy as np
import faiss
import nltk
import cv2
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer, util
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import fadein, fadeout

app = Flask(__name__)



# ── Paths ─────────────────────────────────────────────────────────
# Get the directory of the current script (src) and then its parent
SRC_PATH        = os.path.dirname(os.path.abspath(__file__))
BASE_PATH       = os.path.dirname(SRC_PATH)

COMBINED_CSV    = os.path.join(SRC_PATH, "combined_data.csv")
EMBEDDINGS_PATH = os.path.join(SRC_PATH, "combined_embeddings.npy")
FAISS_INDEX_PATH= os.path.join(SRC_PATH, "combined_faiss.index")
UCF101_CSV      = os.path.join(SRC_PATH, "ucf101_captions.csv")

# These might still need adjustment depending on where the user puts datasets
VIDEOS_PATH     = os.path.join(BASE_PATH, "datasets-project", "data", "MSRVTT", "MSRVTT", "videos", "all")
ANNOTATION_FILE = os.path.join(BASE_PATH, "datasets-project", "data", "MSRVTT", "MSRVTT", "annotation", "MSR_VTT.json")
TEMP_PATH       = os.path.join(BASE_PATH, "temp")
OUTPUT_PATH     = os.path.join(BASE_PATH, "output")

# FFmpeg path - user might need to change this
FFMPEG_PATH     = os.environ.get("FFMPEG_PATH", "ffmpeg") # Default to system PATH

# ── Load Dataset ──────────────────────────────────────────────────
print("Loading dataset...")
merged_df          = pd.read_csv(COMBINED_CSV)
caption_embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
ucf101_df          = pd.read_csv(UCF101_CSV)
model              = SentenceTransformer('all-MiniLM-L6-v2')

# ── Load FAISS Index ──────────────────────────────────────────────
print("Loading FAISS index...")
if os.path.exists(FAISS_INDEX_PATH):
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"  FAISS index loaded — {faiss_index.ntotal:,} vectors ✓")
else:
    faiss_index = None
    print("  FAISS index not found — will use NumPy fallback")

timestamp_lookup = {}
try:
    with open(ANNOTATION_FILE, "r") as f:
        raw_data = json.load(f)
    for video in raw_data['images']:
        vid_id = video['id']
        timestamp_lookup[vid_id] = {
            'start': video.get('start time', 0),
            'end'  : video.get('end time', 10)
        }
    print(f"  Annotation file loaded — {len(timestamp_lookup):,} timestamps ✓")
except FileNotFoundError:
    print("  ⚠  MSR_VTT.json not found — using default clip windows (0–10s)")
except Exception as e:
    print(f"  ⚠  Could not load annotation file: {e} — using defaults")



nlp_model = None  # spaCy not used (incompatible with Python 3.14)

current_scenes = []

# ── HTML ──────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AnimaStudio</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet"/>
<style>
:root{--pink:#e91e8c;--pink2:#ff4db2;--pink3:#c0166f;--teal:#00c9a7;--purple:#7c3aed;--bg:#0a0a12;--bg2:#11111e;--bg3:#181828;--border:rgba(255,255,255,0.07);--text:#f0f0ff;--muted:#6060a0;}
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Plus Jakarta Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column;}
.topbar{background:var(--bg2);border-bottom:1px solid var(--border);padding:0 32px;height:60px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100;}
.logo{display:flex;align-items:center;gap:10px;}
.logo-icon{width:36px;height:36px;background:linear-gradient(135deg,var(--pink3),var(--pink),var(--pink2));border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 4px 16px rgba(233,30,140,0.4);}
.logo-name{font-family:'Space Grotesk',sans-serif;font-size:17px;font-weight:700;background:linear-gradient(135deg,#fff 40%,var(--pink2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.gbadge{margin-left:auto;display:flex;align-items:center;gap:6px;background:linear-gradient(135deg,rgba(124,58,237,0.15),rgba(233,30,140,0.15));border:1px solid rgba(124,58,237,0.3);border-radius:20px;padding:5px 14px;font-size:11px;font-weight:700;color:#a78bfa;}
.avatar{width:34px;height:34px;background:linear-gradient(135deg,var(--pink3),var(--purple));border-radius:9px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;}
.layout{flex:1;display:grid;grid-template-columns:1fr 1fr;height:calc(100vh - 60px);overflow:hidden;}
.lpanel{background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow-y:auto;}
.rpanel{background:var(--bg);display:flex;flex-direction:column;overflow-y:auto;}
.ph{padding:24px 28px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;}
.picon{width:38px;height:38px;border-radius:11px;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.picon.pink{background:linear-gradient(135deg,var(--pink3),var(--pink));}
.picon.purple{background:linear-gradient(135deg,#5b21b6,var(--purple));}
.picon .material-icons-round{font-size:20px;color:#fff;}
.ptitle{font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;}
.psub{font-size:11px;color:var(--muted);margin-top:2px;}
.pb{padding:24px 28px;flex:1;}
.lbl{font-size:11px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;color:var(--muted);margin-bottom:8px;display:flex;align-items:center;gap:5px;}
.lbl .material-icons-round{font-size:13px;}
.storybox{width:100%;background:var(--bg3);border:1px solid var(--border);border-radius:12px;padding:14px 16px;color:var(--text);font-family:inherit;font-size:13.5px;line-height:1.7;resize:none;height:220px;outline:none;transition:all 0.2s;margin-bottom:20px;}
.storybox:focus{border-color:rgba(233,30,140,0.5);box-shadow:0 0 0 3px rgba(233,30,140,0.08);}
.storybox::placeholder{color:rgba(96,96,160,0.5);}
.durbox{background:var(--bg3);border:1px solid var(--border);border-radius:12px;padding:14px 18px;margin-bottom:20px;display:flex;align-items:center;justify-content:space-between;}
.durlbl{font-size:11px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;color:var(--muted);display:flex;align-items:center;gap:6px;}
.durlbl .material-icons-round{font-size:13px;}
.durinp{background:transparent;border:none;outline:none;color:var(--text);font-family:inherit;font-size:22px;font-weight:800;width:70px;text-align:right;}
.dursuf{color:var(--muted);font-size:12px;margin-left:5px;}
.genbtn{width:100%;padding:16px;background:linear-gradient(135deg,var(--pink3),var(--pink),var(--pink2));color:#fff;border:none;border-radius:12px;font-family:inherit;font-size:15px;font-weight:700;cursor:pointer;transition:all 0.3s;display:flex;align-items:center;justify-content:center;gap:10px;box-shadow:0 4px 24px rgba(233,30,140,0.4);margin-bottom:20px;}
.genbtn:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(233,30,140,0.5);}
.genbtn:disabled{opacity:0.5;cursor:not-allowed;transform:none;}
.genbtn .material-icons-round{font-size:20px;}
.statbar{padding:12px 16px;border-radius:10px;font-size:13px;display:none;align-items:center;gap:10px;margin-bottom:16px;}
.statbar.loading{display:flex;background:rgba(255,165,0,0.08);border:1px solid rgba(255,165,0,0.2);color:#ffb347;}
.statbar.success{display:flex;background:rgba(0,201,167,0.08);border:1px solid rgba(0,201,167,0.2);color:var(--teal);}
.statbar.error{display:flex;background:rgba(255,50,80,0.08);border:1px solid rgba(255,50,80,0.2);color:#ff6680;}
.spin{width:15px;height:15px;border:2px solid rgba(255,179,71,0.25);border-top-color:#ffb347;border-radius:50%;animation:spin 0.8s linear infinite;flex-shrink:0;}
@keyframes spin{to{transform:rotate(360deg);}}
.pipe{border-top:1px solid var(--border);padding-top:20px;}
.pipetitle{font-size:11px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;color:var(--muted);margin-bottom:14px;display:flex;align-items:center;gap:6px;}
.steps{display:flex;flex-direction:column;gap:10px;}
.step{display:flex;align-items:center;gap:12px;}
.sdot{width:26px;height:26px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;flex-shrink:0;transition:all 0.3s;}
.sdot.todo{background:var(--bg3);color:var(--muted);border:1px solid var(--border);}
.sdot.active{background:linear-gradient(135deg,var(--pink3),var(--pink));animation:puls 1.5s infinite;}
.sdot.done{background:linear-gradient(135deg,#009e82,var(--teal));}
@keyframes puls{0%,100%{box-shadow:0 0 0 0 rgba(233,30,140,0.5);}50%{box-shadow:0 0 0 5px rgba(233,30,140,0);}}
.sinfo{flex:1;}
.sname{font-size:12px;font-weight:600;color:var(--text);}
.sbar{height:3px;background:rgba(255,255,255,0.06);border-radius:10px;margin-top:5px;overflow:hidden;}
.sfill{height:100%;border-radius:10px;background:linear-gradient(90deg,var(--pink),var(--purple));transition:width 0.5s ease;}
.vsec{flex:1;display:flex;flex-direction:column;}
.vhdr{padding:24px 28px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}
.vtrow{display:flex;align-items:center;gap:12px;}
.rbadge{padding:4px 12px;background:rgba(0,201,167,0.12);border:1px solid rgba(0,201,167,0.25);border-radius:20px;font-size:10px;font-weight:700;color:var(--teal);display:none;}
.rbadge.show{display:block;}
.vwrap{flex:1;display:flex;align-items:center;justify-content:center;padding:28px;min-height:280px;}
.vph{width:100%;max-width:560px;aspect-ratio:16/9;background:var(--bg2);border:2px dashed rgba(233,30,140,0.2);border-radius:16px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;color:var(--muted);}
.vph .material-icons-round{font-size:52px;color:rgba(233,30,140,0.25);}
.vph p{font-size:14px;font-weight:600;}
.vph span{font-size:12px;color:rgba(96,96,160,0.7);}
video{width:100%;max-width:560px;border-radius:16px;border:1px solid var(--border);box-shadow:0 8px 40px rgba(0,0,0,0.5);display:none;}
.vacts{padding:0 28px 24px;display:none;gap:12px;}
.vacts.show{display:flex;}
.abtn{flex:1;padding:12px;border-radius:11px;border:none;font-family:inherit;font-size:13px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:8px;transition:all 0.2s;}
.abtn.dl{background:linear-gradient(135deg,#009e82,var(--teal));color:#fff;}
.abtn.rs{background:var(--bg2);border:1px solid var(--border);color:var(--muted);}
.abtn:hover{transform:translateY(-1px);}
.abtn .material-icons-round{font-size:18px;}
.vinfo{padding:16px 28px;border-top:1px solid var(--border);display:none;gap:24px;flex-wrap:wrap;}
.vinfo.show{display:flex;}
.iitem{display:flex;flex-direction:column;gap:2px;}
.ival{font-size:14px;font-weight:700;font-family:'Space Grotesk',sans-serif;}
.ikey{font-size:10px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:0.5px;}
.scenesec{padding:20px 28px;border-top:1px solid var(--border);display:none;}
.scenesec.show{display:block;}
.scenetitle{font-size:11px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;color:var(--muted);margin-bottom:14px;}
.scenegrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:12px;}
.scenecard{background:var(--bg2);border:1px solid var(--border);border-radius:12px;overflow:hidden;animation:fup 0.3s ease both;transition:all 0.2s;}
.scenecard:hover{border-color:rgba(233,30,140,0.4);transform:translateY(-2px);}
@keyframes fup{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}
.sceneph{width:100%;aspect-ratio:16/9;background:var(--bg3);display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:22px;}
.scenebot{padding:8px 10px;}
.scenenum{font-size:10px;font-weight:800;color:var(--pink2);margin-bottom:2px;}
.scenetxt{font-size:11px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.stag{display:inline-block;margin-top:3px;padding:1px 6px;border-radius:20px;font-size:9px;font-weight:700;background:rgba(124,58,237,0.15);color:#a78bfa;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.08);border-radius:10px;}
@media(max-width:900px){.layout{grid-template-columns:1fr;height:auto;}}
</style>
</head>
<body>
<header class="topbar">
  <div class="logo">
    <div class="logo-icon">&#127916;</div>
    <span class="logo-name">AnimaStudio</span>
  </div>
  <div class="gbadge">&#128194; Dataset-Powered Animation</div>
  <div class="avatar">V</div>
</header>

<div class="layout">

  <div class="lpanel">
    <div class="ph">
      <div class="picon pink"><span class="material-icons-round">auto_awesome</span></div>
      <div>
        <div class="ptitle">Story Input</div>
        <div class="psub">Paste your story — matched to real video clips</div>
      </div>
    </div>
    <div class="pb">

      <label class="lbl"><span class="material-icons-round">edit</span> Your Story</label>
      <textarea class="storybox" id="story"
        placeholder="Write your story here...&#10;&#10;Example: A hero picks up a sword. He fights the villain bravely. Even though the villain is strong, the hero wins."></textarea>

      <label class="lbl"><span class="material-icons-round">timer</span> Duration</label>
      <div class="durbox">
        <span class="durlbl"><span class="material-icons-round">hourglass_empty</span> Video length</span>
        <div style="display:flex;align-items:baseline;">
          <input class="durinp" type="number" id="duration" value="30" min="10" max="120"/>
          <span class="dursuf">seconds</span>
        </div>
      </div>

      <button class="genbtn" id="genBtn" onclick="generateVideo()">
        <span class="material-icons-round">play_circle</span> Generate Animation
      </button>

      <div class="statbar" id="statBar">
        <div class="spin" id="spin"></div>
        <span id="statTxt"></span>
      </div>

      <div class="pipe">
        <div class="pipetitle"><span class="material-icons-round">linear_scale</span> Pipeline</div>
        <div class="steps">
          <div class="step">
            <div class="sdot todo" id="d1">1</div>
            <div class="sinfo">
              <div class="sname">Story Split + Dataset Match</div>
              <div class="sbar"><div class="sfill" id="b1" style="width:0%"></div></div>
            </div>
          </div>
          <div class="step">
            <div class="sdot todo" id="d2">2</div>
            <div class="sinfo">
              <div class="sname">Video Clip Extraction</div>
              <div class="sbar"><div class="sfill" id="b2" style="width:0%"></div></div>
            </div>
          </div>
          <div class="step">
            <div class="sdot todo" id="d3">3</div>
            <div class="sinfo">
              <div class="sname">Style + Emotion Grading</div>
              <div class="sbar"><div class="sfill" id="b3" style="width:0%"></div></div>
            </div>
          </div>
          <div class="step">
            <div class="sdot todo" id="d4">4</div>
            <div class="sinfo">
              <div class="sname">Final Stitch</div>
              <div class="sbar"><div class="sfill" id="b4" style="width:0%"></div></div>
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>

  <div class="rpanel">
    <div class="vsec">
      <div class="vhdr">
        <div class="vtrow">
          <div class="picon purple"><span class="material-icons-round">smart_display</span></div>
          <div>
            <div class="ptitle">Output Video</div>
            <div class="psub">Real video clips matched to your story</div>
          </div>
        </div>
        <div class="rbadge" id="rbadge">&#10003; Ready</div>
      </div>

      <div class="vwrap">
        <div class="vph" id="vph">
          <span class="material-icons-round">movie_creation</span>
          <p>No video yet</p>
          <span>Write a story and click Generate</span>
        </div>
        <video id="vplayer" controls></video>
      </div>

      <div class="vacts" id="vacts">
        <button class="abtn dl" onclick="downloadVideo()">
          <span class="material-icons-round">download</span> Download
        </button>
        <button class="abtn rs" onclick="resetAll()">
          <span class="material-icons-round">refresh</span> New Story
        </button>
      </div>

      <div class="vinfo" id="vinfo">
        <div class="iitem"><div class="ival" id="iScenes">-</div><div class="ikey">Scenes</div></div>
        <div class="iitem"><div class="ival" id="iDur">-</div><div class="ikey">Duration</div></div>
        <div class="iitem"><div class="ival">1280x720</div><div class="ikey">Resolution</div></div>
        <div class="iitem"><div class="ival">Dataset</div><div class="ikey">Source</div></div>
      </div>
    </div>

    <div class="scenesec" id="sceneSec">
      <div class="scenetitle">&#127917; Matched Scenes</div>
      <div class="scenegrid" id="sceneGrid"></div>
    </div>
  </div>

</div>

<script>
function ss(n, s, p) {
  let d = document.getElementById('d' + n), b = document.getElementById('b' + n);
  d.className = 'sdot ' + s;
  d.textContent = s === 'done' ? '\u2713' : n;
  b.style.width = p + '%';
}
function rs() { [1,2,3,4].forEach(n => ss(n,'todo',0)); }
function stat(t, x) {
  let b = document.getElementById('statBar');
  b.className = 'statbar ' + t;
  document.getElementById('statTxt').textContent = x;
  document.getElementById('spin').style.display = t === 'loading' ? 'block' : 'none';
}

async function generateVideo() {
  const story = document.getElementById('story').value.trim();
  const dur   = parseInt(document.getElementById('duration').value);
  const btn   = document.getElementById('genBtn');
  if (!story) { stat('error', 'Please enter your story first!'); return; }

  btn.disabled = true; rs();
  document.getElementById('vplayer').style.display = 'none';
  document.getElementById('vph').style.display = 'flex';
  document.getElementById('vacts').classList.remove('show');
  document.getElementById('vinfo').classList.remove('show');
  document.getElementById('sceneSec').classList.remove('show');
  document.getElementById('rbadge').classList.remove('show');
  document.getElementById('sceneGrid').innerHTML = '';

  stat('loading', 'Matching story to dataset clips...');
  ss(1,'active',40); setTimeout(()=>ss(1,'active',80),1500);

  try {
    const res  = await fetch('/generate', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ story, duration: dur })
    });
    const data = await res.json();

    if (data.success) {
      ss(1,'done',100);
      setTimeout(()=>ss(2,'done',100), 200);
      setTimeout(()=>ss(3,'done',100), 400);
      setTimeout(()=>ss(4,'done',100), 600);
      stat('success', 'Animation ready! ' + data.message);

      const vp = document.getElementById('vplayer');
      vp.src = '/video?t=' + Date.now();
      vp.style.display = 'block';
      document.getElementById('vph').style.display = 'none';
      document.getElementById('vacts').classList.add('show');
      document.getElementById('vinfo').classList.add('show');
      document.getElementById('rbadge').classList.add('show');
      document.getElementById('iScenes').textContent = (data.scenes?.length || 0) + ' clips';
      document.getElementById('iDur').textContent    = dur + 's';

      if (data.scenes?.length) {
        document.getElementById('sceneGrid').innerHTML = data.scenes.map((s,i) => `
          <div class="scenecard" style="animation-delay:${i*0.06}s">
            <div class="sceneph">&#127916;</div>
            <div class="scenebot">
              <div class="scenenum">SCENE ${i+1}</div>
              <div class="scenetxt">${s.sentence}</div>
              <span class="stag">&#128194; Dataset</span>
            </div>
          </div>`).join('');
        document.getElementById('sceneSec').classList.add('show');
      }
    } else {
      stat('error', data.error || 'Generation failed'); rs();
    }
  } catch(e) {
    stat('error', e.message); rs();
  }
  btn.disabled = false;
}

function downloadVideo() { window.location.href = '/download'; }
function resetAll() {
  document.getElementById('story').value = '';
  document.getElementById('vplayer').style.display = 'none';
  document.getElementById('vph').style.display = 'flex';
  document.getElementById('vacts').classList.remove('show');
  document.getElementById('vinfo').classList.remove('show');
  document.getElementById('sceneSec').classList.remove('show');
  document.getElementById('rbadge').classList.remove('show');
  document.getElementById('statBar').className = 'statbar';
  document.getElementById('sceneGrid').innerHTML = '';
  rs();
  document.getElementById('story').focus();
}
</script>
</body>
</html>"""


# ── Scene Graph Extractor (smarter dataset search) ────────────────
def extract_scene_graph(sentence):
    """Extract subject/action/object to build a better search query"""
    if nlp_model is None:
        return sentence
    try:
        doc      = nlp_model(sentence)
        subject  = ""
        action   = ""
        obj      = ""
        location = ""
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass") and not subject:
                subject = token.lemma_
            if token.pos_ == "VERB" and not action:
                action = token.lemma_
            if token.dep_ in ("dobj", "pobj") and not obj:
                obj = token.lemma_
            if token.dep_ == "pobj" and token.head.text in (
                    "in","at","near","through","on","inside","outside","across"):
                location = token.lemma_
        parts    = [p for p in [subject, action, obj, location] if p]
        enhanced = " ".join(parts) if parts else sentence
        print(f"    → scene graph: [{subject}]→[{action}]→[{obj}] loc:[{location}]")
        return enhanced
    except Exception:
        return sentence


# ── Smart Sentence Splitter ───────────────────────────────────────
def split_into_scenes(story_text):
    """Split story into short chunks so we get more scenes"""
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)
    raw = nltk.sent_tokenize(story_text)
    scenes = []
    for sent in raw:
        words = sent.split()
        if len(words) <= 8:
            scenes.append(sent)
        elif len(words) <= 16:
            mid = len(words) // 2
            scenes.append(" ".join(words[:mid]))
            scenes.append(" ".join(words[mid:]))
        else:
            t1 = len(words) // 3
            t2 = 2 * len(words) // 3
            scenes.append(" ".join(words[:t1]))
            scenes.append(" ".join(words[t1:t2]))
            scenes.append(" ".join(words[t2:]))
    print(f"  Story split: {len(raw)} sentences → {len(scenes)} scenes")
    return scenes


# ── Ollama Query Enhancer ─────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"

def ollama_enhance_query(sentence):
    """
    Send story sentence to local Ollama (gemma3:4b).
    Returns a richer search query for FAISS.
    Falls back to original sentence if Ollama is not running.
    """
    prompt = (
        f"Convert this story sentence into 10-15 descriptive keywords "
        f"for searching a video dataset. Focus on actions, objects, setting, "
        f"and visual details. Return ONLY the keywords, no explanation, no punctuation.\n\n"
        f"Story sentence: \"{sentence}\"\n\nKeywords:"
    )
    try:
        payload = json.dumps({
            "model"  : OLLAMA_MODEL,
            "prompt" : prompt,
            "stream" : False
        }).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw      = resp.read().decode("utf-8")
            result   = json.loads(raw)
            enhanced = result.get("response", "").strip()
            enhanced = " ".join(enhanced.split())
            if enhanced:
                return enhanced
    except Exception:
        # Silently fall back to original sentence for a clean output
        pass
    return sentence


# ── Batch 5: Smart Clip Picker — keyword scoring ─────────────────
ACTION_KEYWORDS = {
    'fight' : ['fight','battle','combat','attack','punch','kick','sword','clash','duel','strike','warrior','soldier'],
    'run'   : ['run','chase','flee','sprint','dash','escape','rush','race','hurry'],
    'win'   : ['win','victory','triumph','celebrate','defeat','success','champion','conquer'],
    'pick'  : ['pick','grab','hold','take','lift','wield','carry','raise','grip'],
    'think' : ['think','plan','idea','clever','smart','strategy','mind','decide','wonder'],
    'walk'  : ['walk','move','stroll','march','approach','enter','leave','travel','step'],
    'fall'  : ['fall','drop','collapse','stumble','slip','crash','tumble'],
    'stand' : ['stand','wait','watch','look','observe','guard','stay','remain'],
}

def score_caption_keyword(caption, sentence):
    """
    Score a dataset caption against the story sentence
    by counting shared action keyword matches.
    Returns int — higher = better visual match.
    """
    sl  = sentence.lower()
    cl  = caption.lower()
    score = 0
    for category, keywords in ACTION_KEYWORDS.items():
        sent_hit    = any(k in sl for k in keywords)
        caption_hit = any(k in cl for k in keywords)
        if sent_hit and caption_hit:
            score += 3   # strong match — same action category
        elif caption_hit:
            score += 1   # weak match — caption has action but sentence doesn't
    return score


# ── Batch 12: Character Tracker ───────────────────────────────────
CHARACTER_ROLES = {
    'hero'    : ['hero','protagonist','warrior','knight','soldier','fighter','champion','brave'],
    'villain' : ['villain','enemy','antagonist','dark','evil','monster','demon','foe','beast'],
    'crowd'   : ['people','crowd','group','team','army','soldiers','citizens','everyone'],
    'animal'  : ['animal','dog','cat','horse','bird','beast','creature','lion','tiger'],
}

def detect_characters(sentence):
    """Detect which character roles appear in this sentence."""
    sl = sentence.lower()
    found = []
    for role, keywords in CHARACTER_ROLES.items():
        if any(k in sl for k in keywords):
            found.append(role)
    return found  # e.g. ['hero', 'villain']

def character_boost(caption, characters):
    """
    Boost caption score if it visually matches the character roles.
    Hero scenes → prefer person/human clips
    Villain scenes → prefer darker/combat clips
    """
    if not characters:
        return 0
    cl = caption.lower()
    boost = 0
    if 'hero' in characters and any(w in cl for w in ['person','man','woman','people','human','player']):
        boost += 2
    if 'villain' in characters and any(w in cl for w in ['fight','attack','dark','combat','battle','enemy']):
        boost += 2
    if 'crowd' in characters and any(w in cl for w in ['group','crowd','people','team','many','audience']):
        boost += 2
    if 'animal' in characters and any(w in cl for w in ['animal','dog','cat','horse','bird','running']):
        boost += 2
    return boost


# ── Dataset Matching — FAISS + Ollama + Smart Picker + Characters ─
def match_story_to_videos(story_text):
    scenes = split_into_scenes(story_text)
    used, matched = [], []

    # ── Check Ollama once at start ────────────────────────────────
    ollama_available = False
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags", method="GET"
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            ollama_available = resp.status == 200
    except Exception:
        pass
    # Ollama status check silenced for clean output

    for i, sentence in enumerate(scenes):
        context    = " ".join(scenes[max(0,i-2):i]) if i > 0 else ""
        enhanced   = extract_scene_graph(sentence)
        boost      = "action adventure animation cartoon colorful scene clip video"
        characters = detect_characters(sentence)   # Batch 12

        if characters:
            print(f"    [Characters] Detected: {characters}")

        # ── Ollama query enhancement ──────────────────────────────
        if ollama_available:
            ollama_query = ollama_enhance_query(sentence)
            query = f"{ollama_query} {boost}".strip()
        else:
            query = f"{context} {enhanced} {boost}".strip()

        # Encode query → 384-dim vector
        emb = model.encode([query]).astype("float32")
        t0  = time.time()

        if faiss_index is not None:
            # ── FAISS search → top-50 candidates ─────────────────
            faiss.normalize_L2(emb)
            D, I      = faiss_index.search(emb, 50)
            search_ms = (time.time() - t0) * 1000

            # ── Batch 5: Smart Picker — score all candidates ──────
            best_idx   = None
            best_score = -1

            for rank, idx in enumerate(I[0]):
                if idx in used:
                    continue
                candidate_row     = merged_df.iloc[int(idx)]
                candidate_caption = str(candidate_row.get('caption', ''))

                # Combine: FAISS similarity + keyword score + character boost
                faiss_sim  = float(D[0][rank])
                kw_score   = score_caption_keyword(candidate_caption, sentence)
                char_boost = character_boost(candidate_caption, characters)
                total      = faiss_sim * 10 + kw_score + char_boost

                if total > best_score:
                    best_score = total
                    best_idx   = int(idx)

            if best_idx is None:
                best_idx = int(I[0][0])   # absolute fallback

            score  = float(D[0][0])
            method = "FAISS+Ollama+Smart" if ollama_available else "FAISS+Smart"

        else:
            # ── NumPy fallback ────────────────────────────────────
            sims = util.cos_sim(emb, caption_embeddings)[0].numpy()
            for idx in used:
                sims[idx] *= 0.05
            best_idx  = int(np.argmax(sims))
            score     = float(sims[best_idx])
            search_ms = (time.time() - t0) * 1000
            method    = "NumPy"

        row = merged_df.iloc[best_idx]
        used.append(best_idx)
        matched.append({
            'sentence'  : sentence,
            'video_id'  : row['video_id'],
            'caption'   : row['caption'],
            'source'    : row.get('source', 'msrvtt'),
            'score'     : score,
            'characters': characters
        })
        print(f"  Scene {i+1}: {sentence[:45]}")
        print(f"    → {row['video_id']} | score={score:.4f} | {method} {search_ms:.1f}ms")
        print(f"    → caption: {str(row['caption'])[:60]}")

    return matched
    return matched


# ── Style Processing ──────────────────────────────────────────────
def apply_style(frame, style):
    frame = cv2.resize(frame, (1280,720), interpolation=cv2.INTER_LANCZOS4)
    if style in ('cartoon','pixar'):
        lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        l     = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(l)
        frame = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
        sm    = frame
        for _ in range(4): sm = cv2.bilateralFilter(sm,9,75,75)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(cv2.medianBlur(gray,7),255,
                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,2)
        res   = cv2.bitwise_and(sm, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        hsv   = cv2.cvtColor(res, cv2.COLOR_BGR2HSV).astype("float32")
        h,s,v = cv2.split(hsv)
        s=np.clip(s*2.5,0,255); v=np.clip(v*1.2,0,255)
        res   = cv2.cvtColor(cv2.merge([h,s,v]).astype("uint8"), cv2.COLOR_HSV2BGR)
    elif style == 'anime':
        sm = frame
        for _ in range(6): sm = cv2.bilateralFilter(sm,9,50,50)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(cv2.medianBlur(gray,5),255,
                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,7,2)
        res   = cv2.bitwise_and(sm, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        hsv   = cv2.cvtColor(res, cv2.COLOR_BGR2HSV).astype("float32")
        h,s,v = cv2.split(hsv)
        s     = np.clip(s*3.0,0,255)
        res   = cv2.cvtColor(cv2.merge([h,s,v]).astype("uint8"), cv2.COLOR_HSV2BGR)
    elif style == 'vivid':
        lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        l     = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8)).apply(l)
        frame = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
        sm    = frame
        for _ in range(3): sm = cv2.bilateralFilter(sm,9,100,100)
        hsv   = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV).astype("float32")
        h,s,v = cv2.split(hsv)
        s=np.clip(s*2.8,0,255); v=np.clip(v*1.3,0,255)
        res   = cv2.cvtColor(cv2.merge([h,s,v]).astype("uint8"), cv2.COLOR_HSV2BGR)
    elif style == 'sketch':
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv   = cv2.bitwise_not(gray)
        blur  = cv2.GaussianBlur(inv,(21,21),0)
        sk    = cv2.divide(gray, cv2.bitwise_not(blur), scale=256)
        res   = cv2.cvtColor(sk, cv2.COLOR_GRAY2BGR)
    else:
        res = frame
    return cv2.filter2D(res, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))


# ── Emotion Color Grading ─────────────────────────────────────────
def detect_emotion(sentence):
    """Return emotion label for a sentence — used for logging once per clip."""
    sl = sentence.lower()
    if any(w in sl for w in ['win','won','victory','celebrate','triumph','success','cheer']):
        return 'joy/victory'
    elif any(w in sl for w in ['fight','battle','attack','villain','enemy',
                                'sword','slash','punch','kick','combat','duel']):
        return 'fight/tension'
    elif any(w in sl for w in ['run','flee','escape','chase','danger',
                                'rush','sprint','dash','hurry']):
        return 'urgency/chase'
    elif any(w in sl for w in ['pick','grab','hold','take','lift',
                                'carry','wield','raise','prepare','ready']):
        return 'build-up'
    elif any(w in sl for w in ['think','plan','intelligence','clever',
                                'smart','strategy','idea','mind']):
        return 'thinking'
    return 'neutral'


def apply_emotion_grade(frame, sentence):
    """Apply color grade to a single frame — no print, called per frame."""
    sl = sentence.lower()

    if any(w in sl for w in ['win','won','victory','celebrate','triumph','success','cheer']):
        b, g, r = cv2.split(frame)
        r = cv2.add(r, 25)
        g = cv2.add(g, 15)
        b = cv2.subtract(b, 15)
        frame = cv2.merge([b, g, r])
        frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=10)

    elif any(w in sl for w in ['fight','battle','attack','villain','enemy',
                                'sword','slash','punch','kick','combat','duel']):
        b, g, r = cv2.split(frame)
        r = cv2.add(r, 35)
        g = cv2.subtract(g, 10)
        b = cv2.subtract(b, 20)
        frame = cv2.merge([b, g, r])
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=-15)

    elif any(w in sl for w in ['run','flee','escape','chase','danger',
                                'rush','sprint','dash','hurry']):
        b, g, r = cv2.split(frame)
        b = cv2.add(b, 30)
        r = cv2.subtract(r, 15)
        frame = cv2.merge([b, g, r])
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=-20)

    elif any(w in sl for w in ['pick','grab','hold','take','lift',
                                'carry','wield','raise','prepare','ready']):
        frame = cv2.convertScaleAbs(frame, alpha=0.85, beta=-20)
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype("float32")
        h, s, v = cv2.split(hsv)
        s     = np.clip(s * 0.75, 0, 255)
        frame = cv2.cvtColor(
            cv2.merge([h, s, v]).astype("uint8"), cv2.COLOR_HSV2BGR)

    elif any(w in sl for w in ['think','plan','intelligence','clever',
                                'smart','strategy','idea','mind']):
        b, g, r = cv2.split(frame)
        b = cv2.add(b, 20)
        g = cv2.add(g, 15)
        r = cv2.subtract(r, 10)
        frame = cv2.merge([b, g, r])
        frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)

    else:
        frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)

    return frame


# ══════════════════════════════════════════════════════════════════
# RICH SCENE GENERATOR — Pure PIL + OpenCV, No API, No GPU
# Detects environment + action + emotion from sentence text
# Draws: sky, ground, environment, two characters, effects, caption
# ══════════════════════════════════════════════════════════════════
def generate_comic_panel(sentence, style, index):
    W, H  = 1280, 720
    sl    = sentence.lower()

    # ── Load fonts ────────────────────────────────────────────────
    try:
        font_lg = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf", 30)
        font_sm = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf",   22)
        font_xs = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf",   18)
    except Exception:
        font_lg = ImageFont.load_default()
        font_sm = font_lg
        font_xs = font_lg

    # ── Detect environment from keywords ─────────────────────────
    if any(w in sl for w in ['forest','tree','wood','jungle','nature','park']):
        env = 'forest'
    elif any(w in sl for w in ['castle','palace','throne','kingdom','dungeon','tower']):
        env = 'castle'
    elif any(w in sl for w in ['city','street','building','urban','town','road']):
        env = 'city'
    elif any(w in sl for w in ['mountain','cliff','hill','rock','cave','valley']):
        env = 'mountain'
    elif any(w in sl for w in ['sea','ocean','beach','river','lake','water','ship']):
        env = 'ocean'
    elif any(w in sl for w in ['night','dark','moon','star','midnight','shadow']):
        env = 'night'
    else:
        env = 'plains'

    # ── Detect action ─────────────────────────────────────────────
    if any(w in sl for w in ['fight','sword','battle','attack','punch','kick','slash','combat','duel','strike']):
        action = 'fight'
    elif any(w in sl for w in ['run','chase','flee','escape','rush','sprint','dash']):
        action = 'run'
    elif any(w in sl for w in ['win','won','victory','triumph','celebrate','success','defeat']):
        action = 'victory'
    elif any(w in sl for w in ['pick','grab','hold','take','lift','wield','raise']):
        action = 'pick'
    elif any(w in sl for w in ['think','plan','clever','smart','intelligence','strategy','idea']):
        action = 'think'
    else:
        action = 'stand'

    # ── Style colour palettes ─────────────────────────────────────
    palettes = {
        'cartoon': dict(
            sky1=(255,220,80),  sky2=(255,140,40),
            ground=(80,160,60), ground2=(50,120,40),
            char1=(30,30,30),   char2=(80,20,20),
            accent=(220,50,50), border=(20,20,20),
            caption_bg=(255,255,230), caption_fg=(10,10,10),
            vignette=(180,80,0)
        ),
        'anime': dict(
            sky1=(130,160,255), sky2=(200,100,220),
            ground=(60,140,80), ground2=(40,100,60),
            char1=(30,30,80),   char2=(100,20,80),
            accent=(255,80,180),border=(20,20,80),
            caption_bg=(230,230,255), caption_fg=(20,20,80),
            vignette=(60,0,120)
        ),
        'vivid': dict(
            sky1=(255,60,120),  sky2=(60,210,255),
            ground=(50,200,100),ground2=(30,160,70),
            char1=(80,0,80),    char2=(200,50,0),
            accent=(50,230,120),border=(80,0,80),
            caption_bg=(255,240,255), caption_fg=(60,0,60),
            vignette=(150,0,100)
        ),
        'sketch': dict(
            sky1=(230,230,220), sky2=(190,190,180),
            ground=(150,150,140),ground2=(120,120,110),
            char1=(30,30,30),   char2=(60,60,60),
            accent=(90,90,80),  border=(20,20,20),
            caption_bg=(255,255,250), caption_fg=(10,10,10),
            vignette=(80,80,70)
        ),
        'pixar': dict(
            sky1=(60,190,255),  sky2=(255,160,60),
            ground=(70,170,80), ground2=(50,130,60),
            char1=(20,60,120),  char2=(120,40,20),
            accent=(255,190,30),border=(20,60,120),
            caption_bg=(240,250,255), caption_fg=(20,60,120),
            vignette=(0,80,160)
        ),
    }
    p = palettes.get(style, palettes['cartoon'])

    img  = Image.new('RGB', (W, H), p['sky1'])
    draw = ImageDraw.Draw(img)

    # ══ LAYER 1 — Sky gradient ════════════════════════════════════
    sky_h = int(H * 0.58)
    for y in range(sky_h):
        t  = y / sky_h
        r  = int(p['sky1'][0] + (p['sky2'][0]-p['sky1'][0])*t)
        g  = int(p['sky1'][1] + (p['sky2'][1]-p['sky1'][1])*t)
        b  = int(p['sky1'][2] + (p['sky2'][2]-p['sky1'][2])*t)
        draw.line([(0,y),(W,y)], fill=(r,g,b))

    # ══ LAYER 2 — Sun / Moon ──────────────────────────────────────
    if env == 'night':
        # Moon
        draw.ellipse([(W-160,30),(W-80,110)],
                     fill=(240,240,200), outline=(200,200,150), width=2)
        draw.ellipse([(W-140,38),(W-100,78)],
                     fill=p['sky1'])  # crescent cutout
        # Stars
        random.seed(index * 7)
        for _ in range(60):
            sx = random.randint(0, W)
            sy = random.randint(0, sky_h-40)
            sr = random.randint(1, 3)
            draw.ellipse([(sx-sr,sy-sr),(sx+sr,sy+sr)], fill=(255,255,220))
    else:
        # Sun with rays
        sx, sy = 120, 90
        draw.ellipse([(sx-55,sy-55),(sx+55,sy+55)],
                     fill=(255,240,80), outline=(255,200,40), width=3)
        for ang in range(0,360,30):
            ra   = math.radians(ang)
            x1   = int(sx + math.cos(ra)*65)
            y1   = int(sy + math.sin(ra)*65)
            x2   = int(sx + math.cos(ra)*88)
            y2   = int(sy + math.sin(ra)*88)
            draw.line([(x1,y1),(x2,y2)], fill=(255,220,60), width=4)

        # Clouds
        random.seed(index * 3)
        for ci in range(3):
            cx = random.randint(200, W-200)
            cy = random.randint(30, 130)
            for blob in [(-40,0,55,38),(0,-18,50,34),(40,0,55,38),(-20,10,45,30),(20,10,45,30)]:
                draw.ellipse([(cx+blob[0], cy+blob[1]),
                              (cx+blob[0]+blob[2], cy+blob[1]+blob[3])],
                             fill=(255,255,255))

    # ══ LAYER 3 — Background environment ─────────────────────────
    ground_y = sky_h

    if env == 'forest':
        # Rolling hills
        for y in range(ground_y, H):
            t  = (y-ground_y)/(H-ground_y)
            r  = int(p['ground'][0] + (p['ground2'][0]-p['ground'][0])*t)
            g  = int(p['ground'][1] + (p['ground2'][1]-p['ground'][1])*t)
            b  = int(p['ground'][2] + (p['ground2'][2]-p['ground'][2])*t)
            draw.line([(0,y),(W,y)], fill=(r,g,b))
        # Background trees (far)
        for tx in range(0, W, 80):
            th  = random.randint(120, 200)
            ty  = ground_y - th + 20
            tw  = random.randint(55, 80)
            draw.rectangle([(tx+tw//2-8, ty+th//3),(tx+tw//2+8, ty+th)],
                           fill=(80,50,20))
            draw.ellipse([(tx, ty),(tx+tw, ty+th//2)],
                         fill=(30,120,40))
        # Foreground grass tufts
        for gx in range(0, W, 40):
            gy = ground_y + random.randint(0,15)
            draw.polygon([(gx,gy),(gx+8,gy-20),(gx+16,gy)], fill=(50,160,50))

    elif env == 'castle':
        # Ground
        for y in range(ground_y, H):
            draw.line([(0,y),(W,y)], fill=(100,90,80))
        # Castle wall background
        wall_y = ground_y - 180
        draw.rectangle([(0,wall_y),(W,ground_y)], fill=(140,120,100))
        # Battlements
        for bx in range(0, W, 60):
            draw.rectangle([(bx,wall_y-40),(bx+35,wall_y)], fill=(130,110,90))
        # Castle towers
        for tx in [80, W-130]:
            draw.rectangle([(tx, wall_y-200),(tx+100, ground_y)], fill=(120,100,85))
            draw.rectangle([(tx-15,wall_y-240),(tx+115,wall_y-200)], fill=(110,90,75))
            for btx in range(tx-15, tx+115, 35):
                draw.rectangle([(btx,wall_y-270),(btx+22,wall_y-240)], fill=(110,90,75))
        # Gate arch
        draw.rectangle([(W//2-50, ground_y-130),(W//2+50, ground_y)], fill=(30,20,15))
        draw.ellipse([(W//2-50, ground_y-180),(W//2+50, ground_y-80)], fill=(30,20,15))
        # Stone texture lines
        for sy2 in range(wall_y, ground_y, 25):
            draw.line([(0,sy2),(W,sy2)], fill=(110,95,80), width=1)
        for sx2 in range(0, W, 50):
            draw.line([(sx2,wall_y),(sx2,ground_y)], fill=(110,95,80), width=1)

    elif env == 'night':
        for y in range(ground_y, H):
            t  = (y-ground_y)/(H-ground_y)
            r  = int(20 + 30*t); g = int(20+30*t); b = int(40+50*t)
            draw.line([(0,y),(W,y)], fill=(r,g,b))
        # Dark trees silhouettes
        for tx in range(0,W,100):
            th = random.randint(150,280)
            ty = ground_y-th
            draw.rectangle([(tx+35,ty+th//2),(tx+55,ground_y)], fill=(15,15,25))
            draw.ellipse([(tx,ty),(tx+90,ty+th//2+20)], fill=(15,15,25))

    elif env == 'ocean':
        # Water gradient
        for y in range(ground_y, H):
            t  = (y-ground_y)/(H-ground_y)
            r  = int(30+20*t); g = int(100+50*t); b = int(180+50*t)
            draw.line([(0,y),(W,y)], fill=(r,g,b))
        # Wave lines
        for wy in range(ground_y+20, H, 25):
            for wx in range(0, W-30, 60):
                draw.arc([(wx,wy-8),(wx+40,wy+8)], 180, 0,
                         fill=(100,180,220), width=3)
        # Beach / shore
        draw.rectangle([(0,ground_y-15),(W,ground_y+30)], fill=(210,190,130))

    elif env == 'mountain':
        for y in range(ground_y, H):
            draw.line([(0,y),(W,y)], fill=(90,80,70))
        # Mountain peaks
        for peak in [(200,ground_y-280,500),(600,ground_y-350,400),(1000,ground_y-260,380)]:
            px, ph, pw = peak
            draw.polygon([(px-pw//2,ground_y),(px,ground_y-ph),(px+pw//2,ground_y)],
                         fill=(120,110,100), outline=(100,90,80))
            # Snow cap
            draw.polygon([(px-pw//6,ground_y-ph+80),(px,ground_y-ph),
                           (px+pw//6,ground_y-ph+80)], fill=(240,240,250))

    elif env == 'city':
        for y in range(ground_y, H):
            draw.line([(0,y),(W,y)], fill=(80,80,90))
        # Buildings background
        random.seed(index*5)
        for bx in range(0,W,70):
            bh = random.randint(100,250)
            bw = random.randint(50,65)
            by = ground_y - bh
            bc = (random.randint(60,110),random.randint(60,110),random.randint(70,130))
            draw.rectangle([(bx,by),(bx+bw,ground_y)], fill=bc, outline=(40,40,50))
            # Windows
            for wy2 in range(by+15, ground_y-15, 28):
                for wx2 in range(bx+10, bx+bw-10, 18):
                    wc = (240,220,100) if random.random()>0.3 else (30,30,40)
                    draw.rectangle([(wx2,wy2),(wx2+10,wy2+16)], fill=wc)
        # Street / road
        draw.rectangle([(0,ground_y+10),(W,H)], fill=(60,60,65))
        draw.line([(0,ground_y+35),(W,ground_y+35)], fill=(240,220,60), width=4)

    else:
        # Plains — simple rolling ground
        for y in range(ground_y, H):
            t  = (y-ground_y)/(H-ground_y)
            r  = int(p['ground'][0]+(p['ground2'][0]-p['ground'][0])*t)
            g  = int(p['ground'][1]+(p['ground2'][1]-p['ground'][1])*t)
            b  = int(p['ground'][2]+(p['ground2'][2]-p['ground'][2])*t)
            draw.line([(0,y),(W,y)], fill=(r,g,b))
        # Flowers
        for fx in range(30, W, 55):
            fy = ground_y + random.randint(5,30)
            draw.ellipse([(fx-6,fy-6),(fx+6,fy+6)], fill=(255,200,50))
            draw.ellipse([(fx-3,fy-3),(fx+3,fy+3)], fill=(255,255,100))

    # ══ LAYER 4 — Action-specific background effects ──────────────
    if action == 'fight':
        # Dramatic red speed lines from clash point
        clash_x, clash_y = W//2 + 80, ground_y - 220
        for ang in range(0,360,8):
            ra   = math.radians(ang)
            x_e  = int(clash_x + math.cos(ra)*W)
            y_e  = int(clash_y + math.sin(ra)*H)
            alpha_col = tuple(min(255,c+60) for c in p['accent'])
            draw.line([(clash_x,clash_y),(x_e,y_e)],
                      fill=p['accent'], width=2)
        # Impact burst
        for burst_r in [40,30,20]:
            draw.ellipse([(clash_x-burst_r, clash_y-burst_r),
                          (clash_x+burst_r, clash_y+burst_r)],
                         outline=(255,255,100), width=3)

    elif action == 'victory':
        # Celebration sparkles and beams
        for ang in range(0,360,20):
            ra  = math.radians(ang)
            x1  = int(W//2 + math.cos(ra)*100)
            y1  = int(ground_y-280 + math.sin(ra)*100)
            x2  = int(W//2 + math.cos(ra)*220)
            y2  = int(ground_y-280 + math.sin(ra)*220)
            draw.line([(x1,y1),(x2,y2)], fill=p['accent'], width=4)
        # Confetti dots
        random.seed(index*11)
        for _ in range(40):
            cx2 = random.randint(100,W-100)
            cy2 = random.randint(50, ground_y-50)
            cr  = random.randint(5,12)
            cc  = (random.randint(150,255),random.randint(100,255),random.randint(50,200))
            draw.ellipse([(cx2-cr,cy2-cr),(cx2+cr,cy2+cr)], fill=cc)

    elif action == 'run':
        # Speed lines from left
        for ly in range(ground_y-350, ground_y, 22):
            llen = random.randint(80,200)
            draw.line([(0,ly),(llen,ly)], fill=p['accent'], width=2)

    # ══ LAYER 5 — Character 1 (Hero — left side) ─────────────────
    def draw_character(cx, cy, col, scale=1.0, facing='right', pose='stand', is_hero=True):
        s  = scale
        hr = int(26*s)  # head radius

        # Cloak / cape (behind body for hero)
        if is_hero and pose in ('fight','victory','stand'):
            cape_pts = [
                (int(cx),        int(cy-60*s)),
                (int(cx-45*s),   int(cy+80*s)),
                (int(cx-15*s),   int(cy+100*s)),
                (int(cx+10*s),   int(cy+60*s)),
            ]
            cape_col = tuple(max(0,c-40) for c in col)
            draw.polygon(cape_pts, fill=cape_col)

        # Body (torso rectangle — more solid than line)
        draw.rectangle([
            (int(cx-12*s), int(cy-58*s)),
            (int(cx+12*s), int(cy+18*s))
        ], fill=col)

        # Head
        draw.ellipse([
            (int(cx-hr),   int(cy-58*s-hr*2)),
            (int(cx+hr),   int(cy-58*s))
        ], fill=col)

        # Hair / helmet details for hero
        if is_hero:
            draw.ellipse([
                (int(cx-hr-4), int(cy-58*s-hr*2-8)),
                (int(cx+hr+4), int(cy-58*s-4))
            ], outline=p['accent'], width=3)
        else:
            # Villain — horns
            draw.polygon([
                (int(cx-hr+5),  int(cy-58*s-hr*2)),
                (int(cx-hr-10), int(cy-58*s-hr*2-25)),
                (int(cx-hr+15), int(cy-58*s-hr*2-5))
            ], fill=p['accent'])
            draw.polygon([
                (int(cx+hr-5),  int(cy-58*s-hr*2)),
                (int(cx+hr+10), int(cy-58*s-hr*2-25)),
                (int(cx+hr-15), int(cy-58*s-hr*2-5))
            ], fill=p['accent'])

        # Legs
        if pose == 'run':
            draw.line([(int(cx),int(cy+18*s)),(int(cx+40*s),int(cy+90*s))], fill=col, width=int(10*s))
            draw.line([(int(cx),int(cy+18*s)),(int(cx-50*s),int(cy+80*s))], fill=col, width=int(10*s))
        else:
            draw.line([(int(cx),int(cy+18*s)),(int(cx+28*s),int(cy+100*s))], fill=col, width=int(10*s))
            draw.line([(int(cx),int(cy+18*s)),(int(cx-28*s),int(cy+100*s))], fill=col, width=int(10*s))

        # Arms by pose
        if pose == 'fight':
            if facing == 'right':
                # Sword arm forward
                draw.line([(int(cx),int(cy-35*s)),(int(cx+70*s),int(cy-10*s))], fill=col, width=int(10*s))
                # Sword
                draw.line([(int(cx+70*s),int(cy-10*s)),(int(cx+130*s),int(cy-65*s))],
                          fill=(200,200,220), width=int(7*s))
                draw.line([(int(cx+85*s),int(cy-12*s)),(int(cx+95*s),int(cy+5*s))],
                          fill=(180,130,50), width=int(5*s))  # crossguard
                # Shield arm back
                draw.line([(int(cx),int(cy-35*s)),(int(cx-50*s),int(cy+10*s))], fill=col, width=int(10*s))
            else:
                draw.line([(int(cx),int(cy-35*s)),(int(cx-70*s),int(cy-10*s))], fill=col, width=int(10*s))
                draw.line([(int(cx-70*s),int(cy-10*s)),(int(cx-130*s),int(cy-65*s))],
                          fill=(200,200,220), width=int(7*s))
                draw.line([(int(cx-85*s),int(cy-12*s)),(int(cx-95*s),int(cy+5*s))],
                          fill=(180,130,50), width=int(5*s))
                draw.line([(int(cx),int(cy-35*s)),(int(cx+50*s),int(cy+10*s))], fill=col, width=int(10*s))

        elif pose == 'victory':
            draw.line([(int(cx),int(cy-35*s)),(int(cx+60*s),int(cy-90*s))], fill=col, width=int(10*s))
            draw.line([(int(cx),int(cy-35*s)),(int(cx-60*s),int(cy-90*s))], fill=col, width=int(10*s))

        elif pose == 'run':
            draw.line([(int(cx),int(cy-35*s)),(int(cx+55*s),int(cy-60*s))], fill=col, width=int(10*s))
            draw.line([(int(cx),int(cy-35*s)),(int(cx-40*s),int(cy-10*s))], fill=col, width=int(10*s))

        elif pose == 'pick':
            draw.line([(int(cx),int(cy-35*s)),(int(cx+75*s),int(cy-68*s))], fill=col, width=int(10*s))
            draw.line([(int(cx),int(cy-35*s)),(int(cx-40*s),int(cy+5*s))],  fill=col, width=int(10*s))
            # Object being picked up
            draw.rectangle([
                (int(cx+75*s-10), int(cy-80*s)),
                (int(cx+75*s+18), int(cy-52*s))
            ], fill=p['accent'], outline=(200,200,200), width=2)

        elif pose == 'think':
            draw.line([(int(cx),int(cy-35*s)),(int(cx+40*s),int(cy-10*s))], fill=col, width=int(10*s))
            draw.line([(int(cx+40*s),int(cy-10*s)),(int(cx+55*s),int(cy-40*s))], fill=col, width=int(10*s))
            # Thought bubbles
            for tb_r, tb_x, tb_y in [(8,int(cx+75*s),int(cy-60*s)),
                                       (13,int(cx+95*s),int(cy-90*s)),
                                       (20,int(cx+120*s),int(cy-130*s))]:
                draw.ellipse([(tb_x-tb_r,tb_y-tb_r),(tb_x+tb_r,tb_y+tb_r)],
                             outline=col, width=3)
            draw.line([(int(cx),int(cy-35*s)),(int(cx-45*s),int(cy-5*s))],  fill=col, width=int(10*s))

        else:  # stand
            draw.line([(int(cx),int(cy-35*s)),(int(cx+48*s),int(cy-5*s))],  fill=col, width=int(10*s))
            draw.line([(int(cx),int(cy-35*s)),(int(cx-48*s),int(cy-5*s))],  fill=col, width=int(10*s))

    # Character positions based on action
    ground_stand = ground_y - 10
    hero_x   = W//2 - 160
    villain_x= W//2 + 160
    char_y   = ground_stand

    if action == 'fight':
        # Hero left, villain right, closer together
        draw_character(W//2-130, char_y, p['char1'], scale=1.1,
                       facing='right', pose='fight', is_hero=True)
        draw_character(W//2+130, char_y, p['char2'], scale=1.05,
                       facing='left',  pose='fight', is_hero=False)
        # Clash spark at centre
        spark_x, spark_y = W//2, ground_y - 230
        draw.ellipse([(spark_x-22,spark_y-22),(spark_x+22,spark_y+22)],
                     fill=(255,255,150), outline=(255,200,50), width=3)
        for ang2 in range(0,360,45):
            ra2 = math.radians(ang2)
            draw.line([(spark_x,spark_y),
                       (int(spark_x+math.cos(ra2)*38),
                        int(spark_y+math.sin(ra2)*38))],
                      fill=(255,255,100), width=3)

    elif action == 'run':
        draw_character(W//2-60, char_y, p['char1'], scale=1.1,
                       facing='right', pose='run', is_hero=True)
        # Villain chasing behind
        draw_character(W//2-280, char_y, p['char2'], scale=1.0,
                       facing='right', pose='run', is_hero=False)

    elif action == 'victory':
        draw_character(W//2, char_y, p['char1'], scale=1.2,
                       facing='right', pose='victory', is_hero=True)
        # Defeated villain slumped
        draw.ellipse([(W//2+160,char_y-35),(W//2+230,char_y+10)],
                     fill=p['char2'])  # head on ground
        draw.line([(W//2+195,char_y-25),(W//2+280,char_y-5)],
                  fill=p['char2'], width=12)  # body

    elif action == 'pick':
        draw_character(W//2-60, char_y, p['char1'], scale=1.1,
                       facing='right', pose='pick', is_hero=True)

    elif action == 'think':
        draw_character(W//2-60, char_y, p['char1'], scale=1.1,
                       facing='right', pose='think', is_hero=True)

    else:  # stand
        draw_character(W//2-100, char_y, p['char1'], scale=1.1,
                       facing='right', pose='stand', is_hero=True)
        draw_character(W//2+100, char_y, p['char2'], scale=1.0,
                       facing='left',  pose='stand', is_hero=False)

    # ══ LAYER 6 — Vignette (dark edges for cinematic look) ────────
    vig = Image.new('RGB', (W, H), (0,0,0))
    vig_draw = ImageDraw.Draw(vig)
    for vi in range(0, min(W,H)//3, 8):
        alpha = int(180 * (1 - vi/(min(W,H)//3)))
        rc = tuple(max(0, c - alpha//3) for c in p['vignette'])
        vig_draw.rectangle([(vi,vi),(W-vi,H-vi)], outline=rc, width=8)
    img = Image.blend(img, vig, alpha=0.18)
    draw = ImageDraw.Draw(img)

    # ══ LAYER 7 — Caption bar at bottom ───────────────────────────
    cap_h  = 95
    cap_y  = H - cap_h
    draw.rectangle([(0,cap_y),(W,H)], fill=p['caption_bg'])
    draw.line([(0,cap_y),(W,cap_y)], fill=p['border'], width=4)

    # Word wrap caption
    words2, lines2, line2 = sentence.split(), [], ""
    for word in words2:
        test = (line2 + " " + word).strip()
        try:
            bbox = draw.textbbox((0,0), test, font=font_lg)
            tw2  = bbox[2]-bbox[0]
        except Exception:
            tw2 = len(test)*16
        if tw2 < W-80:
            line2 = test
        else:
            if line2: lines2.append(line2)
            line2 = word
    if line2: lines2.append(line2)

    text_y2 = cap_y + 12
    for ln2 in lines2[:2]:
        try:
            bbox = draw.textbbox((0,0), ln2, font=font_lg)
            tw2  = bbox[2]-bbox[0]
        except Exception:
            tw2 = len(ln2)*16
        draw.text(((W-tw2)//2, text_y2), ln2,
                  font=font_lg, fill=p['caption_fg'])
        text_y2 += 38

    # ══ LAYER 8 — Scene badge (top-left) ──────────────────────────
    draw.ellipse([(18,18),(88,88)],
                 fill=p['accent'], outline=p['border'], width=3)
    try:
        nb = draw.textbbox((0,0), str(index+1), font=font_lg)
        nw = nb[2]-nb[0]
        draw.text(((18+88-nw)//2, 30), str(index+1),
                  font=font_lg, fill='white')
    except Exception:
        draw.text((38, 38), str(index+1), font=font_lg, fill='white')

    # Environment label (top-right)
    env_label = f"⬤ {env.upper()}"
    draw.rectangle([(W-160,18),(W-18,52)],
                   fill=p['accent'], outline=p['border'], width=2)
    try:
        eb = draw.textbbox((0,0), env_label, font=font_xs)
        ew = eb[2]-eb[0]
        draw.text((W-160+(140-ew)//2, 24), env_label,
                  font=font_xs, fill='white')
    except Exception:
        draw.text((W-150, 24), env_label, font=font_xs, fill='white')

    # ══ LAYER 9 — Outer border frame ──────────────────────────────
    draw.rectangle([(3,3),(W-3,H-3)], outline=p['border'], width=7)

    path = os.path.join(TEMP_PATH, f"ai_scene_{index}.jpg")
    img.save(path, "JPEG", quality=95)
    print(f"  [Scene] Panel {index+1} | env={env} | action={action} | style={style} ✓")
    return path


# ── Scene Image Generator — 100% offline, no API, no GPU ──────────
# Calls generate_comic_panel (rich scene renderer) directly
def generate_scene_image(sentence, style, index):
    try:
        path = generate_comic_panel(sentence, style, index)
        if path and os.path.exists(path):
            print(f"  [Scene] ✓ Scene {index+1} ready!")
            return path
    except Exception as e:
        print(f"  [Scene] Error: {e}")
    return None


# ── Motion Types ──────────────────────────────────────────────────
MOTIONS = ['zoom_in','zoom_out','pan_right','pan_left','zoom_in','pan_right']

def image_to_clip(image_path, output_path, clip_duration, style, scene_index=0, sentence=""):
    img     = cv2.imread(image_path)
    img     = cv2.resize(img,(1280,720),interpolation=cv2.INTER_LANCZOS4)
    img     = apply_style(img, style)
    img     = apply_emotion_grade(img, sentence)   # ← emotion color grade
    fps     = 30
    nframes = int(clip_duration * fps)
    tmp     = output_path.replace(".mp4","_raw.mp4")
    out     = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280,720))
    h, w    = img.shape[:2]
    motion  = MOTIONS[scene_index % len(MOTIONS)]

    for f in range(nframes):
        prog = f / max(nframes-1, 1)

        if motion == 'zoom_in':
            scale = 1.0 + 0.20 * prog
            nh,nw = int(h/scale), int(w/scale)
            y0,x0 = (h-nh)//2, (w-nw)//2
        elif motion == 'zoom_out':
            scale = 1.20 - 0.20 * prog
            nh,nw = int(h/scale), int(w/scale)
            y0,x0 = (h-nh)//2, (w-nw)//2
        elif motion == 'pan_right':
            nh,nw = int(h*0.88), int(w*0.88)
            y0    = (h-nh)//2
            x0    = int((w-nw)*prog)
        elif motion == 'pan_left':
            nh,nw = int(h*0.88), int(w*0.88)
            y0    = (h-nh)//2
            x0    = int((w-nw)*(1.0-prog))
        else:
            scale = 1.0 + 0.10*prog
            nh,nw = int(h/scale), int(w/scale)
            y0,x0 = (h-nh)//2, (w-nw)//2

        y0    = max(0, min(y0, h-nh))
        x0    = max(0, min(x0, w-nw))
        frame = cv2.resize(img[y0:y0+nh, x0:x0+nw], (1280,720),
                           interpolation=cv2.INTER_LINEAR)

        fade_f = min(20, nframes//4)
        if f < fade_f:
            frame = (frame * (f/fade_f)).astype(np.uint8)
        elif f > nframes-fade_f:
            frame = (frame * ((nframes-f)/fade_f)).astype(np.uint8)

        out.write(frame)

    out.release()
    cmd = [FFMPEG_PATH,"-y","-i",tmp,"-c:v","libx264","-crf","14",
           "-preset","slow","-movflags","+faststart", output_path]
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(tmp): os.remove(tmp)
    return output_path if os.path.exists(output_path) else None


# ── Extract Dataset Clip ──────────────────────────────────────────
def extract_clip(video_id, idx, source='msrvtt'):
    out = os.path.join(TEMP_PATH, f"clip_{idx}.mp4")
    if source == 'ucf101':
        row = ucf101_df[ucf101_df['video_id']==video_id]
        if not row.empty:
            cmd = [FFMPEG_PATH,"-y","-i",row.iloc[0]['path'],"-t","10",
                   "-vf","scale=1280:720:flags=lanczos",
                   "-c:v","libx264","-crf","14","-preset","slow",
                   "-c:a","aac","-b:a","192k", out]
            subprocess.run(cmd, capture_output=True)
            return out if os.path.exists(out) else None
    path = os.path.join(VIDEOS_PATH, f"{video_id}.mp4")
    if os.path.exists(path):
        ts  = timestamp_lookup.get(video_id, {'start':0,'end':10})
        dur = max(ts['end']-ts['start'], 5)
        cmd = [FFMPEG_PATH,"-y","-i",path,
               "-ss",str(ts['start']),"-t",str(dur),
               "-vf","scale=1280:720:flags=lanczos",
               "-c:v","libx264","-crf","14","-preset","slow",
               "-c:a","aac","-b:a","192k", out]
        subprocess.run(cmd, capture_output=True)
        return out if os.path.exists(out) else None
    return None


def style_clip(inp, out, style):
    cap = cv2.VideoCapture(inp)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    tmp = out.replace(".mp4","_tmp.mp4")
    wr  = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280,720))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        wr.write(apply_style(frame, style))
    cap.release(); wr.release()
    cmd = [FFMPEG_PATH,"-y","-i",tmp,"-i",inp,
           "-map","0:v","-map","1:a?",
           "-c:v","libx264","-crf","14","-preset","slow",
           "-c:a","aac","-b:a","192k",
           "-movflags","+faststart","-shortest", out]
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(tmp): os.remove(tmp)
    return out


def apply_emotion_to_video(inp, out, sentence):
    """
    Apply emotion color grade + subtitle text to a real video.
    Reads every frame, applies color grade, writes subtitle,
    outputs final mp4. Pure OpenCV — no API, no GPU.
    """
    emotion = detect_emotion(sentence)
    print(f"    [Emotion] {emotion} grade applied")
    sl  = sentence.lower()
    cap = cv2.VideoCapture(inp)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    tmp = out.replace(".mp4","_emo.mp4")
    wr  = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (1280,720))

    # Subtitle text — wrap to 2 lines max
    words = sentence.split()
    lines, line = [], ""
    for word in words:
        if len(line) + len(word) + 1 <= 65:
            line = (line + " " + word).strip()
        else:
            if line: lines.append(line)
            line = word
    if line: lines.append(line)
    sub_lines = lines[:2]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (1280,720),
                           interpolation=cv2.INTER_LINEAR)

        # Apply emotion color grade
        frame = apply_emotion_grade(frame, sentence)

        # Draw subtitle bar at bottom
        bar_y = 720 - 70
        cv2.rectangle(frame, (0, bar_y), (1280, 720),
                      (0,0,0), -1)
        cv2.rectangle(frame, (0, bar_y), (1280, bar_y+2),
                      (255,255,255), -1)

        # Draw each subtitle line centered
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.65
        thickness  = 2
        y_pos      = bar_y + 28
        for ln in sub_lines:
            (tw, th), _ = cv2.getTextSize(ln, font, font_scale, thickness)
            x_pos = (1280 - tw) // 2
            # Shadow
            cv2.putText(frame, ln, (x_pos+2, y_pos+2),
                        font, font_scale, (0,0,0), thickness+1)
            # White text
            cv2.putText(frame, ln, (x_pos, y_pos),
                        font, font_scale, (255,255,255), thickness)
            y_pos += 30

        wr.write(frame)

    cap.release()
    wr.release()

    # Re-encode with ffmpeg for proper mp4
    cmd = [FFMPEG_PATH,"-y","-i",tmp,
           "-c:v","libx264","-crf","18","-preset","fast",
           "-movflags","+faststart", out]
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(tmp): os.remove(tmp)
    return out if os.path.exists(out) else inp
def stitch(clips, total_dur):
    per = total_dur / len(clips)
    adj = []
    print(f"  Stitching {len(clips)} clips at {per:.1f}s each...")
    for i, cp in enumerate(clips):
        op  = os.path.join(TEMP_PATH, f"adj_{i}.mp4")
        cap = cv2.VideoCapture(cp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frm = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        cur = max(frm/fps if fps>0 else 10, 1.0)
        if cur < per:
            flt = (f"setpts={per/cur}*PTS" if per/cur<=4.0
                   else f"loop={int(per/cur)+1}:size={int(frm)}:start=0,"
                        f"setpts=N/FRAME_RATE/TB,trim=duration={per}")
        else:
            flt = f"setpts={1/min(cur/per,4.0)}*PTS"
        cmd = [FFMPEG_PATH,"-y","-i",cp,"-filter:v",flt,
               "-c:v","libx264","-crf","14","-preset","slow","-an", op]
        subprocess.run(cmd, capture_output=True)
        adj.append(op if os.path.exists(op) else cp)
        print(f"  Clip {i+1} adjusted ({cur:.1f}s → {per:.1f}s)  ✓")

    vclips = []
    for p in adj:
        if os.path.exists(p):
            c = VideoFileClip(p).resize((1280,720))
            c = fadein(c, 0.8); c = fadeout(c, 0.8)
            vclips.append(c)

    final = concatenate_videoclips(vclips, method="compose")
    out_f = os.path.join(OUTPUT_PATH, "animation_output.mp4")

    # Use ffmpeg concat for reliability
    txt = os.path.join(TEMP_PATH, "concat.txt")
    with open(txt, "w") as f:
        for p in adj:
            if os.path.exists(p):
                f.write(f"file '{p}'\n")
    cmd = [FFMPEG_PATH,"-y","-f","concat","-safe","0","-i",txt,
           "-c:v","libx264","-crf","14","-preset","slow",
           "-movflags","+faststart", out_f]
    result = subprocess.run(cmd, capture_output=True)
    if not os.path.exists(out_f):
        # fallback to moviepy
        final.write_videofile(out_f, fps=30, codec="libx264",
                              audio_codec="aac", bitrate="8000k",
                              verbose=False, logger=None)

    sz = os.path.getsize(out_f)/1024/1024 if os.path.exists(out_f) else 0
    print(f"  Final video: {out_f} ({sz:.1f} MB)")
    for c in vclips: c.close()
    final.close()
    return out_f


# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/generate', methods=['POST'])
def generate():
    global current_scenes
    try:
        data     = request.get_json()
        story    = data.get('story','').strip()
        duration = int(data.get('duration', 30))
        style    = data.get('style', 'cartoon')

        if not story:
            return jsonify({'success':False,'error':'Story is empty'})

        print(f"\n{'='*55}")
        print(f"Duration:{duration}s | Style:{style}")
        print(f"Story: {story[:80]}")
        print('='*55)

        for f in os.listdir(TEMP_PATH):
            try: os.remove(os.path.join(TEMP_PATH,f))
            except: pass

        # ── Step 1: Split story + match dataset ───────────────────
        print("\nStep 1: Story split + dataset matching...")
        matches  = match_story_to_videos(story)
        clip_dur = duration / len(matches)

        clips, scene_results = [], []

        # ── Step 2: Extract real dataset video clips ───────────────
        print(f"\nStep 2: Extracting {len(matches)} dataset video clips...")
        for i, m in enumerate(matches):

            # Extract real video from dataset
            cp = extract_clip(m['video_id'], i, m.get('source','msrvtt'))

            if cp and os.path.exists(cp):
                # Apply style transfer (cartoon/anime/sketch) to video
                op = os.path.join(TEMP_PATH, f"styled_{i}.mp4")
                style_clip(cp, op, style)

                # Apply emotion color grade + subtitle overlay
                final_clip = os.path.join(TEMP_PATH, f"scene_{i}.mp4")
                apply_emotion_to_video(op, final_clip, m['sentence'])

                if os.path.exists(final_clip):
                    clips.append(final_clip)
                    scene_results.append({
                        'sentence' : m['sentence'],
                        'video_id' : m['video_id'],
                        'source'   : 'dataset',
                        'motion'   : 'video',
                        'has_image': False
                    })
                    print(f"  Scene {i+1} ready [dataset video] ✓")
                    continue

            # Last resort — rich scene image with motion
            print(f"  Scene {i+1} — dataset missing, using scene panel...")
            img_path = generate_scene_image(m['sentence'], style, i)
            if img_path:
                clip_path = os.path.join(TEMP_PATH, f"scene_{i}.mp4")
                result = image_to_clip(img_path, clip_path,
                                       clip_dur, style, i, m['sentence'])
                if result and os.path.exists(result):
                    clips.append(result)
                    scene_results.append({
                        'sentence' : m['sentence'],
                        'video_id' : m['video_id'],
                        'source'   : 'scene',
                        'motion'   : MOTIONS[i % len(MOTIONS)],
                        'has_image': True
                    })
                    print(f"  Scene {i+1} ready [scene panel fallback]")

        if not clips:
            return jsonify({'success':False,
                            'error':'No clips generated. Check dataset paths.'})

        # ── Step 3+4: Stitch all clips → final video ──────────────
        print(f"\nStep 3+4: Stitching {len(clips)} clips...")
        stitch(clips, duration)

        current_scenes = scene_results
        msg = f"{len(clips)} scenes | {duration}s | {style}"
        print(f"\nDone! {msg}")

        return jsonify({'success'    : True,
                        'message'    : msg,
                        'scenes'     : scene_results,
                        'comic_count': 0})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success':False,'error':str(e)})


@app.route('/scene_image/<int:idx>')
def scene_image(idx):
    if idx < len(current_scenes):
        p = os.path.join(TEMP_PATH, f"ai_scene_{idx}.jpg")
        if os.path.exists(p):
            return send_file(p, mimetype='image/jpeg')
    return "Not found", 404


@app.route('/video')
def video():
    p = os.path.join(OUTPUT_PATH, "animation_output.mp4")
    if not os.path.exists(p): return "Not found", 404
    return send_file(p, mimetype='video/mp4')


@app.route('/download')
def download():
    p = os.path.join(OUTPUT_PATH, "animation_output.mp4")
    return send_file(p, as_attachment=True, download_name="animation_output.mp4")


if __name__ == '__main__':
    app.run(debug=True, port=5000)