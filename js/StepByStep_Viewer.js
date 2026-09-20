import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// ★ 強力なサンセリフ体フォントスタックの定義
const FONT_STACK = "'Source Sans Pro', 'Helvetica Neue', Arial, 'Hiragino Kaku Gothic ProN', 'Hiragino Sans', Meiryo, sans-serif";

// ★ BASE_STYLE 内に font-family を !important 付きで追加
const BASE_STYLE = `
    display:flex;
    flex-direction:column;
    align-items:center;
    padding:10px;
    background:#111;
    border-radius:8px;
    width:100%;
    height:100%;
    box-sizing:border-box;
    overflow:hidden;
    font-family: ${FONT_STACK} !important;
`;

const BLANK_GIF = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
const MIN_INTERVAL_MS = 30;

// ---- スライダーを操作しやすくする ----
// ブラウザ標準のスライダーはつまみが小さく、ステップ数が少ないとつかみにくいため、
// つまみ・トラックを大きくし、マウスホイールでも 1 ステップずつ動かせるようにする。
const SLIDER_CSS = `
.sbs-range { -webkit-appearance:none; appearance:none; width:100%; height:24px; margin:2px 0;
             background:transparent; cursor:pointer; }
.sbs-range::-webkit-slider-runnable-track { height:8px; background:#444; border-radius:4px; }
.sbs-range::-webkit-slider-thumb { -webkit-appearance:none; appearance:none; box-sizing:border-box;
             width:20px; height:20px; margin-top:-6px; border-radius:50%;
             background:#4da3ff; border:2px solid #fff; }
.sbs-range::-moz-range-track { height:8px; background:#444; border-radius:4px; }
.sbs-range::-moz-range-thumb { box-sizing:border-box; width:20px; height:20px; border-radius:50%;
             background:#4da3ff; border:2px solid #fff; }
.sbs-range:hover::-webkit-slider-thumb { background:#7bbcff; }
.sbs-range:focus-visible { outline:1px solid #4da3ff; outline-offset:2px; }
`;

const ensureSliderStyle = () => {
    if (document.getElementById("sbs-slider-style")) return;
    const style = document.createElement("style");
    style.id = "sbs-slider-style";
    style.textContent = SLIDER_CSS;
    document.head.appendChild(style);
};

const setupSlider = (slider, onChange) => {
    slider.classList.add("sbs-range");
    slider.step = 1;
    // ComfyUI のフロントエンドは、ホイールを既定でキャンバスのズームに転送する。
    // data-capture-wheel を付け、かつ要素にフォーカスがあるときだけ転送されなくなるため、
    // マウスを載せた時点でフォーカスし、離れたら外す。
    slider.setAttribute("data-capture-wheel", "true");
    slider.addEventListener("mouseenter", () => slider.focus({ preventScroll: true }));
    slider.addEventListener("mouseleave", () => { if (document.activeElement === slider) slider.blur(); });
    const move = (delta) => {
        const max = parseInt(slider.max) || 0;
        const next = Math.max(0, Math.min(max, (parseInt(slider.value) || 0) + delta));
        if (next !== parseInt(slider.value)) {
            slider.value = next;
            onChange();
        }
    };
    // ホイール: 下=次のステップ / 上=前のステップ
    slider.addEventListener("wheel", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        move(ev.deltaY > 0 ? 1 : -1);
    }, { passive: false });
    // キー操作: ComfyUI 側のキー処理に奪われないよう、ここで処理して伝播を止める
    slider.addEventListener("keydown", (ev) => {
        const max = parseInt(slider.max) || 0;
        let handled = true;
        if (ev.key === "ArrowLeft" || ev.key === "ArrowDown") move(-1);
        else if (ev.key === "ArrowRight" || ev.key === "ArrowUp") move(1);
        else if (ev.key === "Home") move(-max);
        else if (ev.key === "End") move(max);
        else handled = false;
        if (handled) { ev.preventDefault(); ev.stopPropagation(); }
    });
};

// サーバーが temp フォルダに保存した画像の参照 → /view の URL
const refToUrl = (ref, token) => {
    const params = new URLSearchParams({
        filename: ref.filename,
        subfolder: ref.subfolder ?? "",
        type: ref.type ?? "temp",
        t: token,  // 同じファイル名でも実行ごとに再取得させる
    });
    return api.apiURL(`/view?${params.toString()}`);
};

app.registerExtension({
    name: "Comfy.StepFlowPack",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        if (nodeData.name === "StepStepPlayer" || nodeData.name === "StepStepComparer") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                const isPlayer = nodeData.name === "StepStepPlayer";

                const container = document.createElement("div");
                container.style.cssText = BASE_STYLE;

                let imgA, imgB, div, viewPort, imgPlayer;

                if (isPlayer) {
                    imgPlayer = document.createElement("img");
                    imgPlayer.style.cssText = "width:100%; height:100%; object-fit:contain; background:#000; flex-grow:1; min-height:200px;";
                    imgPlayer.src = BLANK_GIF;
                    container.appendChild(imgPlayer);
                } else {
                    viewPort = document.createElement("div");
                    viewPort.style.cssText = "width:100%; flex-grow:1; min-height:200px; position:relative; overflow:hidden; cursor:col-resize; background:#000;";

                    imgB = document.createElement("img");
                    imgB.style.cssText = "width:100%; height:100%; object-fit:contain; position:absolute; top:0; left:0;";

                    imgA = document.createElement("img");
                    imgA.style.cssText = "width:100%; height:100%; object-fit:contain; position:absolute; top:0; left:0; clip-path:inset(0 50% 0 0);";

                    div = document.createElement("div");
                    div.style.cssText = "position:absolute; top:0; bottom:0; left:50%; width:1px; background:rgba(255,255,255,0.4); z-index:10; pointer-events:none; box-shadow: 0 0 2px rgba(0,0,0,0.3);";

                    imgA.src = imgB.src = BLANK_GIF;
                    viewPort.append(imgB, imgA, div);
                    container.appendChild(viewPort);
                }

                const controls = document.createElement("div");
                controls.style.cssText = "width:100%; flex-shrink:0; padding-top:10px;";

                const s1 = document.createElement("input"); s1.type = "range"; s1.style.width = "100%";
                s1.min = 0; s1.max = 0; s1.value = 0;

                const s2 = isPlayer ? null : document.createElement("input");
                if (s2) { s2.type = "range"; s2.style.width = "100%"; s2.min = 0; s2.max = 0; s2.value = 0; }

                // ★ label にも強制的にフォントを適用
                const label = document.createElement("div");
                label.style.cssText = `
                    font-size:12px;
                    color:#ccc;
                    text-align:center;
                    margin:5px 0;
                    font-weight: 600;
                    font-family: ${FONT_STACK} !important;
                `;
                label.innerText = "Waiting for data...";

                controls.appendChild(s1);
                if (s2) controls.appendChild(s2);
                controls.appendChild(label);

                let timer = null;
                const stopTimer = () => {
                    if (timer) { clearInterval(timer); timer = null; }
                    if (this.playBtn) this.playBtn.innerText = "▶ Play";
                };

                if (isPlayer) {
                    const speedRow = document.createElement("div");
                    speedRow.style.cssText = "display:flex; gap:5px; align-items:center; justify-content:center; margin-bottom:5px;";
                    const speedLabel = document.createElement("span");
                    speedLabel.style.cssText = `font-size:10px; color:#aaa; font-family:${FONT_STACK} !important;`;
                    speedLabel.textContent = "ms:";
                    const speedInput = document.createElement("input");
                    speedInput.type = "number"; speedInput.value = 500; speedInput.min = MIN_INTERVAL_MS; speedInput.style.width = "60px";
                    speedRow.append(speedLabel, speedInput);
                    controls.appendChild(speedRow);

                    const playBtn = document.createElement("button");
                    playBtn.innerText = "▶ Play";
                    playBtn.style.cssText = `width:100%; font-family:${FONT_STACK} !important; cursor:pointer;`;
                    controls.appendChild(playBtn);
                    this.playBtn = playBtn;
                    this.speedInput = speedInput;
                }

                container.appendChild(controls);
                this.addDOMWidget("view_widget", "view", container);

                this.images = [];
                this._preload = [];

                const refresh = () => {
                    if (!this.images || this.images.length === 0) return;
                    s1.max = this.images.length - 1;
                    if (s2) s2.max = this.images.length - 1;

                    const currentIdx1 = parseInt(s1.value);
                    const total = this.images.length;

                    if (isPlayer) {
                        imgPlayer.src = this.images[currentIdx1];
                        label.innerText = `STEP: ${currentIdx1 + 1} / ${total}`;
                    } else {
                        const currentIdx2 = parseInt(s2.value);
                        imgA.src = this.images[currentIdx1];
                        imgB.src = this.images[currentIdx2];
                        label.innerText = `A: Step ${currentIdx1 + 1} | B: Step ${currentIdx2 + 1}`;
                    }
                };

                ensureSliderStyle();
                setupSlider(s1, refresh);
                if (s2) setupSlider(s2, refresh);
                s1.oninput = refresh;
                if (s2) s2.oninput = refresh;

                if (isPlayer) {
                    const intervalMs = () => Math.max(MIN_INTERVAL_MS, parseInt(this.speedInput.value) || 500);
                    const startTimer = () => {
                        this.playBtn.innerText = "⏸ Pause";
                        timer = setInterval(() => {
                            s1.value = (parseInt(s1.value) + 1) % this.images.length;
                            refresh();
                        }, intervalMs());
                    };
                    this.playBtn.onclick = () => {
                        if (timer) { stopTimer(); }
                        else if (this.images.length) { startTimer(); }
                    };
                    // 再生中に速度を変えたら、新しい速度で再開する
                    this.speedInput.onchange = () => {
                        if (timer) { clearInterval(timer); startTimer(); }
                    };
                } else {
                    const handleMove = (e) => {
                        const rect = viewPort.getBoundingClientRect();
                        const clientX = e.clientX ?? e.touches?.[0]?.clientX;
                        if (clientX === undefined || clientX === null) return;
                        const p = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
                        div.style.left = `${p}%`;
                        imgA.style.clipPath = `inset(0 ${100-p}% 0 0)`;
                    };
                    viewPort.onmousemove = handleMove;
                    viewPort.ontouchmove = handleMove;
                }

                const handleData = (refs) => {
                    const token = Date.now();
                    this.images = refs.map((r) => refToUrl(r, token));
                    // 切り替え時のちらつきを避けるため先読みしておく
                    this._preload = this.images.map((u) => { const im = new Image(); im.src = u; return im; });
                    s1.max = this.images.length - 1;
                    if (isPlayer) s1.value = 0;
                    else {
                        s2.max = this.images.length - 1;
                        s1.value = 0;
                        s2.value = this.images.length - 1;
                    }
                    refresh();
                };

                this.onExecuted = (message) => {
                    if (Array.isArray(message?.step_images) && message.step_images.length) {
                        handleData(message.step_images);
                    }
                };

                // ノード削除時にタイマーを止める（残ると削除後も動き続ける）
                const onRemoved = this.onRemoved;
                this.onRemoved = function () {
                    stopTimer();
                    this.images = [];
                    this._preload = [];
                    return onRemoved?.apply(this, arguments);
                };
            };
            nodeType.prototype.onAdded = function() { this.size = [400, 500]; };
        }
    }
});
