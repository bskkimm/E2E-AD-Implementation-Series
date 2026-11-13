"""Reproducible editable illustrations and standalone architecture lessons."""

import ast
import base64
import hashlib
import html
import json
import subprocess
import sys
import textwrap
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path("notebooks/assets/architecture")
NOTEBOOK = Path("notebooks/gameformer_e2e_walkthrough.ipynb")
DEFINITIONS = {
    "gameformer/config.py": ["ModelConfig"],
    "gameformer/geometry.py": ["world_to_local", "local_to_world", "wrap_angle"],
    "gameformer/data.py": ["shapes", "validate_sample", "synthetic_sample"],
    "gameformer/model.py": [
        "segment_points",
        "PositionalEncoding",
        "AgentEncoder",
        "LaneEncoder",
        "CrosswalkEncoder",
        "AttentionBlock",
        "FutureEncoder",
        "GaussianHead",
        "InteractionDecoder",
        "GameFormer",
    ],
    "gameformer/losses.py": ["masked_mean", "interaction_potential", "objective"],
    "gameformer/evaluation.py": [
        "decode",
        "diagnostic_totals",
        "official_metric_inputs",
        "official_metrics",
    ],
}

# Each panel contains an actual numerical trace; full and toy dimensions differ.
# Every figure except the conceptual overview uses illustrative, untrained values.
FIGURES = [
    (
        "overview",
        "From a scene to interacting futures",
        "One common scene; multiple possible futures; one shared mode index.",
        [
            "Observe and encode",
            "Agent history H=11",
            "Nearby lanes + crosswalks",
            "Local frame, explicit masks",
            "Scene Transformer: D=256",
        ],
        [
            "Reason about futures",
            "Level 0: mode + agent query",
            "Predict xy, uncertainty, score",
            "Encode previous futures",
            "Level k responds to level k-1",
        ],
        [
            "Learn and select",
            "Supervise every level",
            "P=2 agents, M=6 joint modes",
            "T=80 steps at dt=0.1 s",
            "Select mode using scores",
        ],
        "model.py",
    ),
    (
        "scene-tensors",
        "Coordinates, validity and targets",
        "Full histories: [B,34,11,9]; targets: [B,2,80,2]. Positions are meters.",
        [
            "World observations",
            "Focal origin: (10, 5) m",
            "Focal heading: 90 degrees",
            "World point: (10, 6) m",
            "Validity bit: true",
        ],
        [
            "Translate, then rotate",
            "delta = (0, 1)",
            "R(-90 deg) delta",
            "Local point = (1, 0)",
            "Zero coordinates stay valid",
        ],
        [
            "Keep labels separate",
            "future_valid: [B,P,T]",
            "false is padding, not x=0",
            "No future enters the encoder",
            "Units: m, rad, m/s, seconds",
        ],
        "geometry.py,data.py",
    ),
    (
        "agent-history",
        "History becomes an actor token",
        "Full: eight numeric inputs -> two LSTM layers -> D256, plus a type embedding.",
        [
            "Toy one-channel memory",
            "Previous cell c = 1.0",
            "Forget gate f = 0.8",
            "Input gate i = 0.5",
            "Candidate g = 0.4",
        ],
        [
            "LSTM gate arithmetic",
            "c_new = f*c + i*g",
            "c_new = 0.8 + 0.2 = 1.0",
            "Output gate o = 0.5",
            "h = o*tanh(c_new)",
        ],
        [
            "Current token",
            "h = 0.3808",
            "Toy type embedding = 0.1",
            "h + type = 0.4808",
            "Focal/background LSTMs differ",
        ],
        "model.py",
    ),
    (
        "lane-encoding",
        "Lane geometry and discrete semantics",
        "Full lanes: [B,P,6,100,17] -> point embeddings [B,P,6,100,256].",
        [
            "Lane point fields",
            "Center/left/right: xy, angle",
            "Speed limit: meters/second",
            "Lane, boundary, signal IDs",
            "Legacy offsets are explicit",
        ],
        [
            "Numeric + categorical",
            "Linear maps for geometry",
            "Sum categorical embeddings",
            "Concat -> 512 -> 384 -> D",
            "Toy MLP output: (0.5,0.6)",
        ],
        [
            "Add point position",
            "At p=0: (sin 0,cos 0)",
            "Position slice = (0,1)",
            "Toy token = (0.5,1.6)",
            "Then training-only dropout",
        ],
        "model.py",
    ),
    (
        "crosswalk-encoding",
        "A point MLP for crosswalks",
        "Full: [B,P,4,50,3] -> [B,P,4,50,256]. No lane positional embedding.",
        [
            "Toy input point",
            "(x,y,heading) = (2,-1,0)",
            "Illustrative 2-channel layer",
            "W rows: (1,0,0), (0,-1,0)",
            "Biases = 0",
        ],
        [
            "Apply linear and ReLU",
            "W * point = (2,1)",
            "ReLU(2,1) = (2,1)",
            "Actual MLP widths:",
            "3 -> 64 -> 128 -> 256",
        ],
        [
            "Preserve the point axis",
            "No temporal pooling here",
            "Each point retains a token",
            "Padding is masked explicitly",
            "Segment pooling comes next",
        ],
        "model.py",
    ),
    (
        "map-segments",
        "Masked channel-wise max pooling",
        "Full: lane groups of 10; crosswalk groups of 10 in the released recipe.",
        [
            "Toy three-point group",
            "Point A: (1,4), valid",
            "Point B: (7,2), valid",
            "Padding: (99,99), invalid",
            "Toy width D=2",
        ],
        [
            "Exclude, then reduce",
            "Mask padding to -infinity",
            "Channel 0 winner: B -> 7",
            "Channel 1 winner: A -> 4",
            "Pooled feature = (7,4)",
        ],
        [
            "Mask policy matters",
            "Strict: partial group hidden",
            "Practical: retain valid points",
            "Empty group -> feature (0,0)",
            "Empty group always hidden",
        ],
        "model.py",
    ),
    (
        "scene-attention",
        "Attention reads only visible keys",
        "Scene length = actors + pooled local map; full released context has 114 tokens.",
        [
            "Toy logits and values",
            "Logits: (0, ln 3, masked)",
            "Values: (2,4,99)",
            "Mask means no key access",
            "Toy one-head scalar values",
        ],
        [
            "Softmax and weighted sum",
            "Probabilities: (.25,.75,0)",
            "Read = .25*2 + .75*4",
            "Read = 3.5",
            "Masked 99 contributes zero",
        ],
        [
            "Residual structure",
            "Scene/self: Norm(q + read)",
            "Then Norm(x + FFN(x))",
            "Release cross: Norm(read)",
            "Cross has no q residual",
        ],
        "model.py",
    ),
    (
        "initial-query",
        "Modes and agents create distinct queries",
        "Full level-0 query: [B,P=2,M=6,D=256]. Toy vectors use D=2.",
        [
            "Three query ingredients",
            "History token: (1,2)",
            "Mode embedding: (.5,0)",
            "Agent embedding: (0,-.5)",
            "No previous future at level 0",
        ],
        [
            "Element-wise addition",
            "(1,2) + (.5,0) + (0,-.5)",
            "Query = (1.5,1.5)",
            "Keys/values = scene tokens",
            "Cross-attention reads context",
        ],
        [
            "Initial futures",
            "One head produces Gaussians",
            "Another head produces scores",
            "A mode index spans both agents",
            "No mode-wise self-attention",
        ],
        "model.py",
    ),
    (
        "gaussian-head",
        "Offsets, uncertainty and mode scores",
        "Each stage: [B,P,M,T,4] = mu_x,mu_y,log_sigma_x,log_sigma_y.",
        [
            "Toy selected timestep",
            "Current xy = (10,2) m",
            "Head offset = (3,1) m",
            "Raw log std = (0, ln 2)",
            "Separate scalar score head",
        ],
        [
            "Anchor to current state",
            "Mean = current + offset",
            "mu = (13,3) m",
            "sigma = exp(log sigma)",
            "sigma = (1,2) m",
        ],
        [
            "Keep meanings distinct",
            "Offsets are from current xy",
            "No cumulative-sum decoder",
            "Log std is not variance",
            "Softmax scores across modes",
        ],
        "model.py",
    ),
    (
        "future-encoding",
        "Previous xy becomes a detached future feature",
        "Full input [B,P,M,80,2]; shared future MLP outputs [B,P,M,256].",
        [
            "Toy first future step",
            "Current xy = (0,0)",
            "Next xy = (1,0) m",
            "dt = 0.1 seconds",
            "Size = (4,2,1.5) m",
        ],
        [
            "Kinematic transformation",
            "delta xy = (1,0) m",
            "Velocity = (10,0) m/s",
            "Heading = atan2(0,1) = 0",
            "xy, angle, velocity, size",
        ],
        [
            "Gradient barrier",
            "Detach kinematic inputs",
            "8 -> 64 -> 256 -> time max",
            "Add type embedding",
            "MLP learns; input xy detached",
        ],
        "model.py",
    ),
    (
        "future-interaction",
        "Pool modes, then mix future agents",
        "Full multi-future [B,P,M,256] -> [B,P,256] -> future self-attention.",
        [
            "Toy two-mode features",
            "Scalar features: (2,4)",
            "Mode probabilities: (.25,.75)",
            "Full model has six modes",
            "Toy example has two modes",
        ],
        [
            "Released probability pool",
            "Weighted values: (.5,3)",
            "Source applies mean:",
            "(.5 + 3) / 2 = 1.75",
            "Normalized sum would be 3.5",
        ],
        [
            "Future interaction",
            "Self-attention across agents",
            "Residual + norm + FFN",
            "Shared future MLP at all k",
            "Independent attention at each k",
        ],
        "model.py",
    ),
    (
        "reasoning-mask",
        "Mask the own-future context token",
        "Level-k query = previous content + own multi-future embedding.",
        [
            "Toy context for agent A",
            "Future interaction A",
            "Future interaction B",
            "Original scene tokens",
            "Both futures mixed beforehand",
        ],
        [
            "Cross-attention key mask",
            "A future token: hidden",
            "B future token: visible",
            "Valid scene tokens: visible",
            "Padding tokens: hidden",
        ],
        [
            "Interpret the mask carefully",
            "Applied after self-attention",
            "Own feature enters query",
            "Mixed B may contain A info",
            "Not total causal isolation",
        ],
        "model.py",
    ),
    (
        "full-assembly",
        "One encoder, a hierarchy of decoders",
        "Full release: E6 scene layers; K3 interactions plus initial level 0.",
        [
            "Prepare common context",
            "C=34 observed actors",
            "P=2 agent-local maps",
            "D=256 features",
            "Scene shape: [B,2,114,256]",
        ],
        [
            "Iterate level-k responses",
            "Level 0 -> level 1",
            "Level 1 -> level 2",
            "Level 2 -> level 3",
            "Keep every intermediate output",
        ],
        [
            "Stack all stages",
            "Trajectories:",
            "[4,B,2,6,80,4]",
            "Scores: [4,B,2,6]",
            "Paper K6 is a named recipe",
        ],
        "model.py",
    ),
    (
        "joint-objective",
        "One winning mode for the joint scene",
        "Illustrative two-mode example; actual model uses six shared modes.",
        [
            "Mean agent distances",
            "Mode 0: agent errors (0,10)",
            "Mode 1: agent errors (2,2)",
            "Joint averages: (5,2)",
            "Select mode 1 for both agents",
        ],
        [
            "Diagonal Gaussian NLL",
            "Toy residual: dx=2, dy=0",
            "sigma_x=sigma_y=1",
            "NLL per timestep = 2",
            "Mean + 3 horizons -> 8",
        ],
        [
            "Supervise all levels",
            "Joint CE uses summed logits",
            "Label smoothing = 0.2",
            "Planning adds .1 potential",
            "1 m interaction: 1/(1+1)=.5",
        ],
        "losses.py",
    ),
    (
        "optimizer-update",
        "A real backward pass and AdamW update",
        "Autocast forward is optional; Gaussian loss arithmetic stays in FP32.",
        [
            "Toy one-parameter loss",
            "w=2, L=0.5*w*w",
            "Gradient dL/dw = 2",
            "Toy AdamW learning rate=.1",
            "Weight decay=0, eps ignored",
        ],
        [
            "First AdamW update",
            "Bias-corrected m = 2",
            "Bias-corrected v = 4",
            "Update = .1*2/sqrt(4)",
            "w_new = 1.9",
        ],
        [
            "Actual training step",
            "Sample-weight accumulation",
            "Unscale -> clip norm at 5",
            "AdamW step -> zero gradients",
            "Final partial group is smaller",
        ],
        "engine.py",
    ),
    (
        "decoding-metrics",
        "Inference scores select a shared future",
        "Selection never reads the target; diagnostics and official metrics differ.",
        [
            "Toy three joint scores",
            "Probabilities: (.1,.8,.1)",
            "Select joint mode index 1",
            "Use this mode for all agents",
            "No oracle target selection",
        ],
        [
            "Toy diagnostic distances",
            "Per-time errors: (1,2,3) m",
            "ADE = (1+2+3)/3 = 2 m",
            "FDE = last error = 3 m",
            "Exclude invalid labels",
        ],
        [
            "Official WOMD boundary",
            "80 steps -> 16 at 2 Hz",
            "Use positions at 4::5",
            "Evaluate at 3,5,8 seconds",
            "minADE/minFDE/miss/mAP SDK",
        ],
        "evaluation.py",
    ),
    (
        "checkpoint-resume",
        "Recover at a complete epoch boundary",
        "Infrastructure workflow: an atomic rename publishes a coherent checkpoint.",
        [
            "Save experiment state",
            "Model, AdamW, scheduler",
            "Scaler, RNG, shuffle generator",
            "Epoch/global step, config",
            "MLflow ID, timing boundary",
        ],
        [
            "Commit one checkpoint",
            "Write latest.tmp",
            "Flush and fsync the file",
            "Atomic rename -> latest.pt",
            "Old checkpoint survives partial write",
        ],
        [
            "Resume deterministically",
            "Reject recipe changes",
            "Restore RNG after construction",
            "Continue at the next epoch",
            "No mid-epoch recovery claim",
        ],
        "engine.py",
    ),
    (
        "tracking-eta",
        "Estimate remaining time after a resume",
        "Infrastructure example; toy timestamps and durations, not measured training.",
        [
            "Total and new-session work",
            "Completed: 6 of 10 epochs",
            "This session: 2 epochs / 20s",
            "Prior active duration: 40s",
            "Total wall time: 100s",
        ],
        [
            "Measure remaining work only",
            "New-session rate: .1 epoch/s",
            "Remaining epochs: 10-6 = 4",
            "Remaining time = 4/.1 = 40s",
            "Total active = 40+20 = 60s",
        ],
        [
            "Visible MLflow overview",
            "Percent, active and wall time",
            "Expected finish with timezone",
            "Provisional before 2 epochs",
            "Include validation/save overhead",
        ],
        "tracking.py",
    ),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_code(module, names):
    text = (ROOT / module).read_text()
    definitions = {
        node.name: node
        for node in ast.parse(text).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    chunks = []
    for name in names:
        node = definitions[name]
        start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
        chunks.append("\n".join(text.splitlines()[start - 1 : node.end_lineno]))
    return "\n\n\n".join(chunks)


def draw_figure(spec, destination):
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont

    # Use wheel-bundled fonts and rasterizer, not host Cairo/fontconfig versions.
    # SVG and PNG use the same layout primitives; SVG text stays editable.
    scale = 2
    picture = Image.new("RGB", (1100 * scale, 620 * scale), "#f5f7fb")
    canvas = ImageDraw.Draw(picture)
    font_root = Path(matplotlib.get_data_path()) / "fonts/ttf"

    identifier, title, subtitle, *rest = spec
    panels, source = rest[:3], rest[3]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="620" viewBox="0 0 1100 620">',
        '<rect width="1100" height="620" fill="#f5f7fb"/>',
        '<g font-family="DejaVu Sans" fill="#172336">',
    ]

    def label(x, y, content, size=15, weight="normal", color="#172336"):
        parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(content)}</text>'
        )
        font = ImageFont.truetype(
            str(font_root / ("DejaVuSans-Bold.ttf" if weight == "bold" else "DejaVuSans.ttf")),
            size * scale,
        )
        canvas.text((x * scale, y * scale), content, font=font, fill=color, anchor="ls")

    label(32, 43, title, 25, "bold")
    for i, line in enumerate(textwrap.wrap(subtitle, 124)):
        label(33, 74 + i * 21, line, 14)
    colors = [("#e7eefb", "#2c518c"), ("#fff0d8", "#805100"), ("#e1f4e9", "#24633e")]
    for i, panel in enumerate(panels):
        x = 32 + i * 365
        background, accent = colors[i]
        parts.append(
            f'<rect x="{x}" y="128" width="336" height="362" rx="12" fill="{background}" stroke="{accent}" stroke-width="1.5"/>'
        )
        canvas.rounded_rectangle(
            (x * scale, 128 * scale, (x + 336) * scale, 490 * scale),
            radius=12 * scale,
            fill=background,
            outline=accent,
            width=3,
        )
        label(x + 17, 164, panel[0], 17, "bold", accent)
        y = 207
        for row in panel[1:]:
            for line in textwrap.wrap(row, 34):
                label(x + 17, y, line)
                y += 25
            y += 12
    for x in (371, 736):
        parts.append(
            f'<path d="M{x} 310 H{x + 17} M{x + 10} 304 L{x + 18} 310 L{x + 10} 316" stroke="#536783" stroke-width="2.5" fill="none"/>'
        )
        for points in (((x, 310), (x + 17, 310)), ((x + 10, 304), (x + 18, 310), (x + 10, 316))):
            canvas.line([(px * scale, py * scale) for px, py in points], fill="#536783", width=5)
    label(33, 528, "Blue: inputs   |   Amber: operation   |   Green: result / interpretation", 14)
    label(
        33,
        558,
        "Conceptual overview."
        if identifier == "overview"
        else "Worked toy values explain the computation; they are not trained-model measurements.",
        14,
    )
    label(
        33,
        588,
        f"Source: gameformer/{source.replace(',', ', gameformer/')}  |  Figure: {identifier}",
        12,
    )
    parts.append("</g></svg>")
    svg = "\n".join(parts) + "\n"
    (destination / (identifier + ".svg")).write_text(svg)
    picture.resize((1100, 620), Image.Resampling.LANCZOS).quantize(colors=64).save(
        destination / (identifier + ".png"), optimize=True
    )


def generate_assets(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    inventory = []
    for figure in FIGURES:
        draw_figure(figure, destination)
        identifier, title, subtitle = figure[:3]
        sources = [f"gameformer/{name}" for name in figure[-1].split(",")]
        sources.append("gameformer/teaching.py")
        inventory.append(
            {
                "id": identifier,
                "anchor": identifier,
                "caption": subtitle,
                "alt": title,
                "mime_type": "image/png",
                "png_sha256": digest(destination / (identifier + ".png")),
                "svg_sha256": digest(destination / (identifier + ".svg")),
                "source_sha256": {path: digest(ROOT / path) for path in sources},
                "generation": "shared editable SVG/Pillow primitives; wheel-bundled DejaVu fonts; 2x supersampling; 64-color PNG",
            }
        )
    manifest = {
        "figure_count": len(FIGURES),
        "figures": inventory,
        "versions": {name: version(name) for name in ("matplotlib", "pillow")},
        "font_sha256": {
            name: digest(Path(__import__("matplotlib").get_data_path()) / "fonts/ttf" / name)
            for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
        },
        "font": "DejaVu Sans",
        "width": 1100,
        "height": 620,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_notebook():
    import nbformat as nbf

    manifest = generate_assets(ROOT / ASSETS)
    specs = {item["id"]: item for item in manifest["figures"]}
    cells = []

    def markdown(identifier, text):
        cells.append(nbf.v4.new_markdown_cell(text, id=identifier))

    def code(identifier, text):
        cells.append(nbf.v4.new_code_cell(text, id=identifier))

    def figure(identifier, body):
        entry = specs[identifier]
        markdown(
            identifier,
            body + f"\n\n![{entry['alt']}](attachment:{identifier}.png)\n\n*{entry['caption']}*",
        )
        cells[-1].attachments = {
            identifier + ".png": {
                "image/png": base64.b64encode(
                    (ROOT / ASSETS / (identifier + ".png")).read_bytes()
                ).decode()
            }
        }

    def definitions(identifier, module, names, probe):
        code(identifier + "-definition", source_code(module, names))
        code(identifier + "-probe", probe)

    figure(
        "overview",
        "# GameFormer, from observations to joint futures\n\n"
        "Reimplement the released WOMD joint predictor in plain PyTorch. Level-k means "
        "each stage responds to the previous stage’s predicted futures. The code below "
        "defines the complete model locally, component by component. Planning uses "
        "the same architecture with a different objective.\n\n"
        "Sources: [paper](https://arxiv.org/abs/2303.05760), "
        "[supplement](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Huang_GameFormer_Game-theoretic_Modeling_ICCV_2023_supplemental.pdf), "
        "[pinned release](https://github.com/MCZhi/GameFormer/tree/fcb0d4a0f5cbbcecf69f9b9796366d6f5f2ce128). "
        "The paper uses K6; the released trainer defaults to K3. Synthetic execution "
        "does not reproduce paper results. This notebook uses a smaller named CPU recipe.",
    )
    markdown(
        "setup",
        "## Setup\n\nRequires Python >=3.11, PyTorch, NumPy and Matplotlib. "
        "Default execution is seeded, offline, on CPU. The full release uses D256, "
        "E6, K3, C34, P2, M6, H11 and T80. We retain T80 for the actual objective "
        "but reduce width, depth and context for the probes.",
    )
    code(
        "imports",
        "import math\nfrom dataclasses import dataclass\nfrom typing import cast\n\n"
        "import matplotlib.pyplot as plt\nimport torch\nfrom torch import Tensor, nn\n"
        "from torch.nn import functional as F\n\n"
        'torch.set_num_threads(1)\ntorch.manual_seed(3407)\nprint("CPU:", torch.__version__)',
    )
    code("model-config-definition", source_code("gameformer/config.py", ["ModelConfig"]))
    figure(
        "scene-tensors",
        "## Data, coordinates and explicit masks\n\n"
        r"$p_{local}=R(-\theta_0)(p_{world}-p_0)$; positions are meters and angles radians. "
        "The first target defines the origin and need not be the autonomous vehicle. "
        "`actor_valid`, `lane_valid`, `crosswalk_valid` and `future_valid` distinguish "
        "missing data from legitimate zero coordinates. Future labels are excluded "
        "from model inputs. A scene contains observations [C,H,9], lane points "
        "[P,L,Q,17], crosswalks [P,W,R,3] and targets [P,T,2].",
    )
    code(
        "sample-definition",
        source_code("gameformer/geometry.py", DEFINITIONS["gameformer/geometry.py"])
        + "\n\n"
        + source_code("gameformer/data.py", DEFINITIONS["gameformer/data.py"]),
    )
    code(
        "sample-probe",
        "config = ModelConfig.smoke()\n"
        "samples = [synthetic_sample(config, i, 3407) for i in range(2)]\n"
        "batch = {key: torch.stack([s[key] for s in samples]) for key in samples[0]}\n"
        'assert batch["actor_valid"][0, 0, -1]\n'
        'assert torch.equal(batch["ego_state"][0, -1, :2], torch.zeros(2))\n'
        "print({key: tuple(value.shape) for key, value in batch.items()})",
    )
    figure(
        "agent-history",
        "## Actor history\n\n"
        r"An LSTM updates memory as $c_t=f_t c_{t-1}+i_t g_t$ and $h_t=o_t\tanh(c_t)$. "
        "Two layers encode eight numeric fields; the last hidden state receives a "
        "categorical type embedding. Separate weights encode the focal actor and "
        "all remaining actors. Invalid history rows are zeroed before encoding; "
        "the current validity controls the actor attention mask.",
    )
    definitions(
        "actor",
        "gameformer/model.py",
        ["AgentEncoder"],
        "actor_encoder = AgentEncoder(config.width)\n"
        'assert actor_encoder(batch["ego_state"]).shape == (2, config.width)',
    )
    figure(
        "lane-encoding",
        "## Lane points\n\nNumeric boundary coordinates use linear projections; "
        "discrete IDs use embeddings. Concatenate three 128-channel geometry features, "
        "64 speed channels and 64 summed semantic channels; apply a 512→384→D MLP, "
        r"then sinusoidal point positions $PE_{p,2j}=\sin(p/10000^{2j/D})$. "
        "The source has a lane-column disagreement: compatibility uses 14/15, "
        "whereas corrected semantics use 15/16. The configuration names this choice.",
    )
    definitions(
        "lanes",
        "gameformer/model.py",
        ["PositionalEncoding", "LaneEncoder"],
        'lane_encoder = LaneEncoder(config)\nassert lane_encoder(batch["map_lanes"]).shape[-1] == config.width',
    )
    figure(
        "crosswalk-encoding",
        "## Crosswalk points\n\n"
        "A local MLP maps three numeric fields (x,y,heading) through 64, 128 and D "
        "channels with ReLU activations. It preserves every point until pooling. "
        "No external map encoder or model library is used.",
    )
    definitions(
        "crosswalks",
        "gameformer/model.py",
        ["CrosswalkEncoder"],
        "crosswalk_encoder = CrosswalkEncoder(config.width)\n"
        'assert crosswalk_encoder(batch["map_crosswalks"]).shape[-1] == config.width',
    )
    figure(
        "map-segments",
        "## Reduce map tokens\n\n"
        r"$z_{segment,d}=\max_{q\in valid} z_{q,d}$. "
        "Different channels can select different points. Every pooled token retains "
        "a padding flag (True means hidden). The released policy hides a partly "
        "invalid segment; the practical policy admits any valid point. Entirely "
        "empty groups produce zero features and remain hidden.",
    )
    definitions(
        "segments",
        "gameformer/model.py",
        ["segment_points"],
        "toy = torch.tensor([[[[1., 4.], [7., 2.], [99., 99.]]]])\n"
        "pooled, padding = segment_points(toy, torch.tensor([[[True, True, False]]]), 3, True)\n"
        "torch.testing.assert_close(pooled, torch.tensor([[[7., 4.]]]))\nassert not padding.any()",
    )
    figure(
        "scene-attention",
        "## Context and attention\n\n"
        r"$Attention(Q,K,V)=softmax(QK^T/\sqrt{d}+mask)V$. "
        "Concatenate actor tokens and the target agent’s local map. The scene "
        "Transformer uses post-normalization and residual connections. The released "
        "cross-attention block normalizes the read without adding the query, an "
        "important implementation detail. FFN means a two-layer feature network.",
    )
    definitions(
        "attention",
        "gameformer/model.py",
        ["AttentionBlock"],
        "attention = AttentionBlock(config, self_attention=True)\n"
        "tokens = torch.randn(2, 3, config.width)\n"
        "assert attention(tokens, tokens, torch.zeros(2, 3, dtype=torch.bool)).shape == tokens.shape",
    )
    figure(
        "initial-query",
        "## Level-0 queries\n\n"
        r"$q_{i,m}=context_i+embedding_{agent,i}+embedding_{mode,m}$. "
        "Each agent and mode gets a distinct query, which reads the common scene "
        "through cross-attention. No previous future is available at this stage.",
    )
    code(
        "query-probe",
        "query = torch.tensor([1., 2.]) + torch.tensor([.5, 0.]) + torch.tensor([0., -.5])\n"
        "torch.testing.assert_close(query, torch.tensor([1.5, 1.5]))",
    )
    figure(
        "gaussian-head",
        "## Decode Gaussian components\n\n"
        "A Gaussian mixture represents alternative possible futures. Each mode has "
        "a mean (x,y), diagonal log standard deviations, and a score. The head emits "
        "offsets from the current position, rather than per-step increments. "
        r"$\mu_t=p_{current}+\Delta p_t$ and $\sigma_t=\exp(log\sigma_t)$.",
    )
    definitions(
        "gaussian",
        "gameformer/model.py",
        ["GaussianHead"],
        "head = GaussianHead(config)\n"
        "means, scores = head(torch.randn(2, 2, 6, config.width), torch.zeros(2, 2, 9))\n"
        "assert means.shape == (2, 2, 6, 80, 4) and scores.shape == (2, 2, 6)",
    )
    figure(
        "future-encoding",
        "## Future kinematics and a gradient barrier\n\n"
        r"$v_t=(p_t-p_{t-1})/0.1$. "
        "Append heading and box dimensions to form eight channels. A shared MLP "
        "and time maximum encode every candidate future. The input kinematics "
        "are detached: this path cannot change previous xy through backpropagation. "
        "The future MLP, prior query content and score paths still learn. "
        "The legacy positive-dx clamp is explicit in configuration.",
    )
    definitions(
        "future",
        "gameformer/model.py",
        ["FutureEncoder"],
        "future_encoder = FutureEncoder(config)\n"
        "xy = torch.randn(2, 2, 6, 80, 2, requires_grad=True)\n"
        "encoded = future_encoder(xy, torch.zeros(2, 2, 9))\n"
        "encoded.sum().backward()\nassert xy.grad is None\n"
        "assert future_encoder.mlp[0].weight.grad is not None",
    )
    figure(
        "future-interaction",
        "## Probability pooling and future interaction\n\n"
        r"The release computes $f_i=\frac1M\sum_m softmax(s_i)_m e_{i,m}$. "
        "This has an additional 1/M compared with a normalized expectation. "
        "Preserve it for compatibility. Self-attention mixes agents’ future tokens; "
        "each interaction level owns its attention and Gaussian head, while "
        "the future-feature MLP is shared across levels.",
    )
    definitions(
        "interaction",
        "gameformer/model.py",
        ["InteractionDecoder"],
        "toy_pool = (torch.tensor([2., 4.])*torch.tensor([.25, .75])).mean()\n"
        "assert toy_pool.item() == 1.75\ninteraction = InteractionDecoder(config)",
    )
    figure(
        "reasoning-mask",
        "## Level-k cross-attention\n\n"
        "Append mixed future tokens to the original scene. Add the own multi-future "
        "embedding to prior query content, then read this extended context. Hide "
        "the own-future key after future self-attention, matching the release. "
        "This masking location does not establish complete causal isolation.",
    )
    code(
        "mask-probe",
        "own_future_hidden = torch.eye(config.predicted, dtype=torch.bool)\n"
        "assert own_future_hidden.tolist() == [[True, False], [False, True]]",
    )
    figure(
        "full-assembly",
        "## Assemble the complete model\n\n"
        "All observed actors appear in every agent-local context. Only the first P "
        "actors are decoded. For each level, predicted means and scores feed the "
        "next level, alongside previous query content. Retain all K+1 stages for "
        "deep supervision. Vectorized agent contexts match release operations "
        "in evaluation; training dropout draws need not follow upstream RNG order.",
    )
    definitions(
        "assembly",
        "gameformer/model.py",
        ["GameFormer"],
        "model = GameFormer(config)\noutputs = model(batch)\n"
        'assert outputs["trajectories"].shape == (2, 2, 2, 6, 80, 4)\n'
        'print("Parameters:", sum(p.numel() for p in model.parameters()))',
    )
    figure(
        "joint-objective",
        "## Actual training objectives\n\n"
        "Select one shared joint mode by its displacement errors. For the joint "
        "predictor, use the temporal mean and 3/5/8-second horizons. "
        r"The diagonal NLL is $log\sigma_x+log\sigma_y+\frac12[(d_x/\sigma_x)^2+(d_y/\sigma_y)^2]$. "
        "The printed Eq. 3 repeats x; the correct y term agrees with the released "
        "code. Add twice the joint score cross-entropy with label smoothing .2. "
        "Sum all levels. Missing labels contribute neither error nor denominator.\n\n"
        "Planning uses its released imitation reduction and adds .1 times the "
        "cross-level potential at levels k≥1. The potential is 1/(distance+1) below "
        "3 m. It compares ego to previous neighbors, and neighbors to previous "
        "other neighbors, matching the released directional objective. The "
        "supplement intentionally omits this penalty for joint prediction.",
    )
    definitions(
        "loss",
        "gameformer/losses.py",
        DEFINITIONS["gameformer/losses.py"],
        'components = objective(outputs, batch, "joint")\n'
        'assert torch.isfinite(components["loss"])\n'
        "print({name: float(value.detach()) for name, value in components.items()})",
    )
    figure(
        "optimizer-update",
        "## Backward and one real optimizer update\n\n"
        "Losses are computed in FP32. The following cell performs the actual "
        "objective, backward pass, gradient clipping and AdamW update. Optimizer "
        "state and randomness must later be saved for faithful recovery. "
        "The teaching example uses one CPU batch; the trainer also supports "
        "sample-weighted accumulation and mixed precision.",
    )
    code(
        "update-probe",
        "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)\n"
        "optimizer.zero_grad(set_to_none=True)\n"
        'loss = objective(model(batch), batch, "joint")["loss"]\nloss.backward()\n'
        "gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)\n"
        'optimizer.step()\nassert torch.isfinite(gradient_norm)\nprint("Gradient norm:", float(gradient_norm))',
    )
    figure(
        "decoding-metrics",
        "## Decode and measure\n\n"
        "Sum agent logits, softmax over modes, and select the same highest-score "
        "mode for every agent. Diagnostic ADE is average valid displacement; "
        "FDE uses each agent’s last valid step. Official WOMD metrics require "
        "the SDK, actual labels and the prescribed split. The adapter below "
        "requires full labeled states and explicit validity; it cannot produce "
        "official results merely by renaming a diagnostic.",
    )
    definitions(
        "metrics",
        "gameformer/evaluation.py",
        DEFINITIONS["gameformer/evaluation.py"],
        "model.eval()\nwith torch.no_grad():\n    inference = model(batch)\n"
        "selected, probabilities = decode(inference)\n"
        'print("Selected xy:", selected.shape)\nprint(diagnostic_totals(inference, batch))\n'
        "sdk_layout = official_metric_inputs(inference, batch)\n"
        'assert sdk_layout["prediction_trajectory"].shape == (2, 1, 6, 2, 16, 2)\n'
        "fig, ax = plt.subplots(figsize=(6, 3))\n"
        'for agent in range(2):\n    truth = batch["future"][0, agent]\n'
        '    path = selected[0, agent]\n    ax.plot(truth[:, 0], truth[:, 1], "--", label=f"target {agent}")\n'
        '    ax.plot(path[:, 0], path[:, 1], label=f"random-model prediction {agent}")\n'
        'ax.set(xlabel="local x (m)", ylabel="local y (m)", title="Synthetic diagnostic after one update")\n'
        "ax.legend(fontsize=7)\nfig.tight_layout()\nplt.show()\nplt.close(fig)",
    )
    figure(
        "checkpoint-resume",
        "## Optional real data and recovery\n\n"
        "Infrastructure is separate from the model. `gameformer.engine` provides "
        "data loading, atomic epoch-boundary checkpoints and tracking. These "
        "imports are deliberately opt-in; the architecture above needs no package "
        "import. Read the data guide for practical-preprocessing differences. "
        "Explicit real mode fails if assets are missing. A full training run "
        "requires a dataset, hardware preflight and an authorized budget. "
        "Schema-2 checkpoints keep the sampler independent of worker startup "
        "and reject changed cache manifests before restoring state. Original "
        "config/environment records remain intact; `sessions/` stores every restart. "
        "Exact recovery covers deterministic datasets without random worker augmentation.",
    )
    code(
        "real-opt-in",
        "RUN_REAL = False\nif RUN_REAL:\n"
        "    from pathlib import Path\n\n    from gameformer.config import RunConfig\n"
        "    from gameformer.engine import train\n\n"
        '    real_config = RunConfig.load("configs/practical_joint.json")\n'
        "    if not real_config.train_data or not Path(real_config.train_data).is_dir():\n"
        '        raise FileNotFoundError("Set train_data and validation_data to genuine prepared caches")\n'
        '    train(real_config, "outputs/real-run")',
    )
    figure(
        "tracking-eta",
        "## Tracking and trustworthy remaining time\n\n"
        "MLflow is optional and logs the experiment’s source/config/data identity, "
        "loss components, gradient norms, trajectories and progress. After resume, "
        "measure the current session’s rate and estimate only remaining work. "
        "Include observed validation and checkpoint overhead; report active time "
        "separately from wall time and downtime. Estimates before two completed "
        "epochs are provisional. `progress.json` works with tracking disabled.",
    )
    code(
        "eta-probe",
        "completed, total = 6, 10\nsession_epochs, session_seconds = 2, 20\n"
        "remaining_seconds = (total-completed)/(session_epochs/session_seconds)\n"
        'assert remaining_seconds == 40\nprint("Toy remaining duration:", remaining_seconds, "seconds")',
    )
    markdown(
        "limits",
        "## What this execution establishes\n\n"
        "The model, actual objectives and optimizer update execute on seeded "
        "synthetic CPU data. It establishes no paper metric, real-data parity or "
        "closed-loop planning score. The release omits the 64-mode EM ensemble "
        "and closed-loop WOMD code; nuPlan adds a distinct planning framework. "
        "The source audit records these boundaries and the corrected validity "
        "behavior. Refer to `COMMAND_GUIDE.md` for training, resume, local MLflow, "
        "loader/step/end-to-end benchmarks and bounded tuning.",
    )
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "architecture_definitions": DEFINITIONS,
        },
    )
    path = ROOT / NOTEBOOK
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", str(path)], check=True, capture_output=True
    )
    return path


def verify_assets(regenerate=None):
    import nbformat

    notebook = nbformat.read(ROOT / NOTEBOOK, as_version=4)
    assert (ROOT / NOTEBOOK).stat().st_size < 1024 * 1024, "Notebook exceeds 1 MiB"
    manifest = json.loads((ROOT / ASSETS / "manifest.json").read_text())
    assert len(manifest["figures"]) == len(FIGURES) == manifest["figure_count"] == 18
    assert notebook.metadata["architecture_definitions"] == DEFINITIONS
    anchors = {cell.id: cell for cell in notebook.cells}
    assert len(anchors) == len(notebook.cells), "Duplicate notebook cell IDs"
    declared = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            for node in ast.parse(cell.source).body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    assert node.name not in declared, f"Shadowed definition: {node.name}"
                    declared[node.name] = node
    for module, names in DEFINITIONS.items():
        expected = {node.name: node for node in ast.parse(source_code(module, names)).body}
        for name in names:
            assert ast.dump(declared[name]) == ast.dump(expected[name]), f"Stale definition: {name}"
    attachment_count = sum(len(cell.get("attachments", {})) for cell in notebook.cells)
    assert attachment_count == len(FIGURES)
    for entry in manifest["figures"]:
        identifier = entry["id"]
        cell = anchors[entry["anchor"]]
        assert cell.source.count("attachment:" + identifier + ".png") == 1
        data = base64.b64decode(cell.attachments[identifier + ".png"]["image/png"])
        assert hashlib.sha256(data).hexdigest() == entry["png_sha256"]
        for suffix in ("png", "svg"):
            assert digest(ROOT / ASSETS / (identifier + "." + suffix)) == entry[suffix + "_sha256"]
        for source, sha in entry["source_sha256"].items():
            assert digest(ROOT / source) == sha, f"Stale figure source: {identifier}, {source}"
    assert notebook.cells[0].id == "overview"
    if regenerate is not None:
        generated = generate_assets(regenerate)
        assert generated == manifest, "Renderer output/version changed: inspect before rebuilding"
    # Independent numerical arithmetic that is displayed in the figures.
    assert (0.25 * 2 + 0.75 * 4) / 2 == 1.75
    assert 0.5 * (2**2 + 0**2) * 4 == 8
    assert (10 - 6) / (2 / 20) == 40
    return {"figures": len(FIGURES), "definitions": len(declared)}


def execute_notebook():
    from dataclasses import asdict

    import nbformat
    import torch

    from .config import ModelConfig
    from .losses import objective
    from .model import GameFormer

    notebook = nbformat.read(ROOT / NOTEBOOK, as_version=4)
    namespace = {"__name__": "__gameformer_lesson__"}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{NOTEBOOK}#{cell.id}", "exec"), namespace)
    local = namespace["model"].eval()
    packaged = GameFormer(ModelConfig(**asdict(namespace["config"]))).eval()
    packaged.load_state_dict(local.state_dict(), strict=True)
    batch = namespace["batch"]
    local.zero_grad(set_to_none=True)
    packaged.zero_grad(set_to_none=True)
    output_local, output_package = local(batch), packaged(batch)
    for name in output_local:
        torch.testing.assert_close(output_local[name], output_package[name], atol=1e-6, rtol=1e-6)
    local_loss = namespace["objective"](output_local, batch, "joint")["loss"]
    package_loss = objective(output_package, batch, "joint")["loss"]
    torch.testing.assert_close(local_loss, package_loss)
    local_loss.backward()
    package_loss.backward()
    left, right = dict(local.named_parameters()), dict(packaged.named_parameters())
    assert left.keys() == right.keys()
    for name in left:
        if left[name].grad is None or right[name].grad is None:
            assert left[name].grad is None and right[name].grad is None
        else:
            torch.testing.assert_close(left[name].grad, right[name].grad, atol=1e-6, rtol=1e-6)
    print("Standalone notebook: output and gradient parity passed")


def main():
    import argparse
    import subprocess
    import sys
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "execute"))
    args = parser.parse_args()
    if args.command == "build":
        print(build_notebook())
    elif args.command == "check":
        with tempfile.TemporaryDirectory() as directory:
            print(verify_assets(regenerate=directory))
        subprocess.run([sys.executable, "-m", "gameformer.teaching", "execute"], check=True)
    else:
        execute_notebook()


if __name__ == "__main__":
    main()
