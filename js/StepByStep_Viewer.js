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
                    imgPlayer.src = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
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
                    
                    imgA.src = imgB.src = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
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

                if (isPlayer) {
                    const speedRow = document.createElement("div");
                    speedRow.style.cssText = "display:flex; gap:5px; align-items:center; justify-content:center; margin-bottom:5px;";
                    const speedInput = document.createElement("input");
                    speedInput.type = "number"; speedInput.value = 500; speedInput.style.width = "60px";
                    speedRow.innerHTML = `<span style='font-size:10px; color:#aaa; font-family:${FONT_STACK} !important;'>ms:</span>`;
                    speedRow.appendChild(speedInput);
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
                let timer = null;

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

                s1.oninput = refresh;
                if (s2) s2.oninput = refresh;

                if (isPlayer) {
                    this.playBtn.onclick = () => {
                        if (timer) { clearInterval(timer); timer = null; this.playBtn.innerText = "▶ Play"; }
                        else {
                            if (!this.images.length) return;
                            this.playBtn.innerText = "⏸ Pause";
                            timer = setInterval(() => {
                                s1.value = (parseInt(s1.value) + 1) % this.images.length;
                                refresh();
                            }, this.speedInput.value);
                        }
                    };
                } else {
                    const handleMove = (e) => {
                        const rect = viewPort.getBoundingClientRect();
                        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
                        if (!clientX) return;
                        const p = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
                        div.style.left = `${p}%`;
                        imgA.style.clipPath = `inset(0 ${100-p}% 0 0)`;
                    };
                    viewPort.onmousemove = handleMove;
                    viewPort.ontouchmove = handleMove;
                }

                const handleData = (imgList) => {
                    this.images = imgList;
                    s1.max = this.images.length - 1;
                    if (isPlayer) s1.value = 0;
                    else {
                        s2.max = this.images.length - 1;
                        s1.value = 0;
                        s2.value = this.images.length - 1;
                    }
                    refresh();
                };

                const msgName = isPlayer ? "step_player_update" : "step_comparer_update";
                api.addEventListener(msgName, (e) => handleData(e.detail.images));
                this.onExecuted = (message) => { if (message?.images) handleData(message.images); };
            };
            nodeType.prototype.onAdded = function() { this.size = [400, 500]; };
        }
    }
});
