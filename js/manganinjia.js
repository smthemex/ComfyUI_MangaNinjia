// ComfyUI_MangaNinjia - modern frontend (ComfyUI 0.4x extension API)
// Points are serialized straight into the node's STRING widget. No custom
// HTTP routes, no global node_instances map, no /upload/image side effects.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CANVAS_SIZE = 512;
const EXT_NAME = "Comfy.MangaNinjia.Points";

function makeStyle() {
    const css = `
.mn-points { display:flex; flex-direction:column; gap:6px; width:100%;
  color:var(--fg-color, #ddd); font-family:inherit; font-size:12px; }
.mn-points-row { display:flex; gap:5px; align-items:center; flex-wrap:wrap; }
.mn-points button { flex:0 0 auto; padding:3px 8px; cursor:pointer;
  color:var(--fg-color, #ddd); background:var(--comfy-menu-bg, #353535);
  border:1px solid var(--border-color, #555); border-radius:4px; font-size:11px; }
.mn-points button:hover { background:var(--comfy-menu-bg-hover, #444); }
.mn-points canvas { display:block; width:100%; height:auto; max-height:420px;
  object-fit:contain; background:#606060; border:1px solid var(--border-color,#555);
  border-radius:4px; cursor:crosshair; touch-action:none; }
.mn-points .mn-info { opacity:.8; font-size:11px; line-height:1.4; }
.mn-points input[type=file] { display:none; }
`;
    const el = document.createElement("style");
    el.textContent = css;
    document.head.appendChild(el);
}

function inputImageUrl(node) {
    // Modern ComfyUI: resolve the upstream LoadImage widget and use /view.
    try {
        const input = (node.inputs || []).find((i) => i.name === "image" && i.link != null);
        if (!input) return null;
        const link = app.graph.links[input.link];
        if (!link) return null;
        const src = app.graph.getNodeById(link.origin_id);
        if (!src) return null;
        const w = (src.widgets || []).find((x) => x.name === "image");
        if (!w || !w.value) return null;
        const v = typeof w.value === "string" ? w.value : (w.value.name || "");
        const params = new URLSearchParams({ filename: v, type: "input", subfolder: "" });
        return api.apiURL("/view?" + params.toString());
    } catch (e) {
        return null;
    }
}

function createCanvasWidget(node, jsonWidget) {
    const root = document.createElement("div");
    root.className = "mn-points";

    const row = document.createElement("div");
    row.className = "mn-points-row";
    const btnInput = document.createElement("button");
    btnInput.textContent = "Use input image";
    const btnFile = document.createElement("button");
    btnFile.textContent = "Load image";
    const btnUndo = document.createElement("button");
    btnUndo.textContent = "Undo";
    const btnClear = document.createElement("button");
    btnClear.textContent = "Clear";
    const file = document.createElement("input");
    file.type = "file";
    file.accept = "image/*";
    row.append(btnInput, btnFile, btnUndo, btnClear, file);

    const canvas = document.createElement("canvas");
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    const ctx = canvas.getContext("2d");

    const info = document.createElement("div");
    info.className = "mn-info";
    root.append(row, canvas, info);

    const state = { points: [], bg: null, objectUrl: null };

    function sync() {
        jsonWidget.value = JSON.stringify(state.points.map((p) => ({ x: p.x, y: p.y, number: p.n })));
        info.textContent = `${state.points.length} point(s) - click to add, right click to undo. ` +
            "Keep the same click order on the reference and target nodes.";
        if (jsonWidget.callback) jsonWidget.callback(jsonWidget.value);
        app.graph.setDirtyCanvas(true, true);
        draw();
    }

    function draw() {
        ctx.fillStyle = "#606060";
        ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        if (state.bg) {
            ctx.drawImage(state.bg, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
        } else {
            ctx.fillStyle = "#9a9a9a";
            ctx.font = "16px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("Load / use the input image as a click backdrop", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
        }
        state.points.forEach((p, i) => {
            ctx.fillStyle = "#00bfff";
            ctx.fillRect(p.x - 6, p.y - 6, 12, 12);
            ctx.fillStyle = "#101010";
            ctx.font = "10px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(String(i + 1), p.x, p.y);
        });
    }

    function setBackground(url) {
        if (!url) return;
        const img = new Image();
        img.onload = () => { state.bg = img; draw(); };
        img.onerror = () => { info.textContent = "Could not load the backdrop image."; };
        img.src = url;
    }

    function canvasPoint(ev) {
        const r = canvas.getBoundingClientRect();
        return {
            x: Math.max(0, Math.min(CANVAS_SIZE - 1, Math.round((ev.clientX - r.left) / r.width * CANVAS_SIZE))),
            y: Math.max(0, Math.min(CANVAS_SIZE - 1, Math.round((ev.clientY - r.top) / r.height * CANVAS_SIZE))),
        };
    }

    canvas.addEventListener("pointerdown", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        if (ev.button === 2) { state.points.pop(); sync(); return; }
        if (ev.button !== 0) return;
        const p = canvasPoint(ev);
        p.n = state.points.length;
        state.points.push(p);
        sync();
    });
    canvas.addEventListener("contextmenu", (ev) => ev.preventDefault());

    btnInput.onclick = () => setBackground(inputImageUrl(node));
    btnFile.onclick = () => file.click();
    file.onchange = () => {
        if (!file.files || !file.files[0]) return;
        if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
        state.objectUrl = URL.createObjectURL(file.files[0]);
        setBackground(state.objectUrl);
    };
    btnUndo.onclick = () => { state.points.pop(); sync(); };
    btnClear.onclick = () => { state.points = []; sync(); };

    const restore = () => {
        try {
            const saved = JSON.parse(jsonWidget.value || "[]");
            if (Array.isArray(saved)) {
                state.points = saved.map((p, i) => ({
                    x: Number(p.x) || 0, y: Number(p.y) || 0, n: i,
                }));
            }
        } catch (e) { /* keep empty */ }
        draw();
        info.textContent = `${state.points.length} point(s) - click to add, right click to undo.`;
    };
    restore();

    // DOM widget with serialize:false -> only points_json goes into the prompt.
    // Never override serializeValue: nodeCreated runs BEFORE the graph loader
    // writes widget values, so returning state.points here silently dropped
    // every point restored from a saved workflow (it always submitted "[]").
    node.addDOMWidget("points_canvas", "manganinjia_points", root, { serialize: false });

    const origConfigure = node.onConfigure;
    node.onConfigure = function () {
        const r = origConfigure?.apply(this, arguments);
        restore();  // widget values are available now
        return r;
    };

    const origRemoved = node.onRemoved;
    node.onRemoved = function () {
        if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
        return origRemoved?.apply(this, arguments);
    };

    node.setSize([420, 560]);
    const origResize = node.onResize;
    node.onResize = function (size) {
        const r = origResize?.apply(this, arguments);
        canvas.style.maxHeight = Math.max(180, size[1] - 110) + "px";
        return r;
    };
    setTimeout(() => setBackground(inputImageUrl(node)), 150);
}

app.registerExtension({
    name: EXT_NAME,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MangaNinjiaPoints") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const w = (this.widgets || []).find((x) => x.name === "points_json");
            if (w) {
                makeStyle();
                createCanvasWidget(this, w);
            }
            return r;
        };
    },
});
