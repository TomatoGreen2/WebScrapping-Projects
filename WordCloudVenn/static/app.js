/* app.js
   - Fetches words from a backend API (Postgres/Neon)
   - Uses top 50 by total frequency for the Venn diagram
   - Shows ALL words grouped by source at the bottom
   - Ellipse-based Venn (ovals), shrink-and-retry placement
*/

(() => {
    // --- DOM ---
    const canvas = document.getElementById("vennCanvas");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const reloadBtn = document.getElementById("reloadBtn");

    // Plain list outputs
    const listOmegaEl = document.getElementById("listOmega");
    const listAxialEl = document.getElementById("listAxial"); // keep id, but we'll label it CN
    const listGuardianEl = document.getElementById("listGuardian");

    // --- Configuration ---
    // Your backend base URL (Render FastAPI)
    // You can override at runtime by setting:
    //   window.API_BASE = "https://webscrapping-projects.onrender.com"
    const API_BASE = (window.API_BASE || "https://webscrapping-projects.onrender.com").replace(/\/+$/, "");

    // If you prefer overriding the full words endpoint directly:
    //   window.WORDS_API_URL = "https://webscrapping-projects.onrender.com/api/words"
    const WORDS_API_URL = window.WORDS_API_URL || `${API_BASE}/api/words`;

    const LISTS = [
        { key: "ACS Omega", bit: 1, fill: "rgba(255, 99, 132, 0.22)", stroke: "rgba(255, 99, 132, 0.80)" },
        { key: "ACS C&N", bit: 2, fill: "rgba(54, 162, 235, 0.20)", stroke: "rgba(54, 162, 235, 0.80)" },
        { key: "The Guardian", bit: 4, fill: "rgba(75, 192, 192, 0.18)", stroke: "rgba(75, 192, 192, 0.80)" },
    ];

    const CANVAS_ASPECT = 700 / 1100;

    // TOP words overall for the diagram
    const TOP_N_VENN = 50;

    // Shrink-and-retry parameters
    const SHRINK_TRIES = 6;
    const SHRINK_FACTOR = 0.88;
    const MIN_FONT_SIZE = 11;

    // ---------------------------
    // Utility
    // ---------------------------
    function setStatus(msg) {
        if (statusEl) statusEl.textContent = msg;
    }

    function clamp(n, a, b) {
        return Math.max(a, Math.min(b, n));
    }

    function normalizeWord(w) {
        return String(w || "").trim().replace(/\s+/g, " ").toLowerCase();
    }

    function getCanvasCssSize() {
        const dpr = window.devicePixelRatio || 1;
        return { w: canvas.width / dpr, h: canvas.height / dpr, dpr };
    }

    function num(n) {
        const x = Number(n);
        return Number.isFinite(x) ? x : 0;
    }

    // ---------------------------
    // Canvas sizing
    // ---------------------------
    function fitCanvasToContainer() {
        const dpr = Math.max(1, window.devicePixelRatio || 1);

        const cssWidth = Math.max(300, canvas.clientWidth || 1100);
        const cssHeight = Math.round(cssWidth * CANVAS_ASPECT);

        canvas.style.height = `${cssHeight}px`;
        canvas.width = Math.round(cssWidth * dpr);
        canvas.height = Math.round(cssHeight * dpr);

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    // ---------------------------
    // Data fetch (API → JSON)
    // ---------------------------
    async function fetchWordsFromApi() {
        const res = await fetch(WORDS_API_URL, { cache: "no-store" });
        if (!res.ok) throw new Error(`Fetch ${WORDS_API_URL} failed: ${res.status} ${res.statusText}`);
        const data = await res.json();

        if (!Array.isArray(data)) {
            throw new Error(`Expected JSON array from ${WORDS_API_URL}`);
        }

        // Normalize to {word, omega, cn, guardian}
        return data
            .map((row) => {
                const word = normalizeWord(row.word);
                return {
                    word,
                    omega: num(row.freq_omega),
                    cn: num(row.freq_cn),
                    guardian: num(row.freq_guardian),
                };
            })
            .filter((r) => r.word);
    }

    // ---------------------------
    // Bottom lists (ALL words by source)
    // ---------------------------
    function formatWordList(rows, key) {
        // rows: [{word, omega, cn, guardian}]
        // key: "omega" | "cn" | "guardian"
        const withFreq = rows
            .map((r) => ({ word: r.word, f: r[key] }))
            .filter((x) => x.f > 0);

        // Sort by frequency desc, then alpha
        withFreq.sort((a, b) => (b.f - a.f) || a.word.localeCompare(b.word, undefined, { sensitivity: "base" }));

        // Show "word (freq)" — change to just word if you prefer
        return withFreq.map((x) => `${x.word} (${x.f})`).join(", ");
    }

    function renderWordListsFromDb(rows) {
        if (listOmegaEl) listOmegaEl.textContent = `ACS Omega: ${formatWordList(rows, "omega")}`;
        if (listAxialEl) listAxialEl.textContent = `ACS C&N: ${formatWordList(rows, "cn")}`;
        if (listGuardianEl) listGuardianEl.textContent = `The Guardian: ${formatWordList(rows, "guardian")}`;
    }

    // ---------------------------
    // Build Venn words (top 50 overall)
    // ---------------------------
    function buildVennWords(rows) {
        // compute total + mask
        const enriched = rows.map((r) => {
            const omega = r.omega > 0;
            const cn = r.cn > 0;
            const guardian = r.guardian > 0;

            const mask = (omega ? 1 : 0) | (cn ? 2 : 0) | (guardian ? 4 : 0);
            const total = r.omega + r.cn + r.guardian;

            return { word: r.word, mask, total };
        });

        // Keep words that appear in at least one source
        const nonZero = enriched.filter((e) => e.mask !== 0 && e.total > 0);

        // Top N by total frequency
        nonZero.sort((a, b) => (b.total - a.total) || a.word.localeCompare(b.word, undefined, { sensitivity: "base" }));
        return nonZero.slice(0, TOP_N_VENN);
    }

    // ---------------------------
    // Ellipse Venn geometry + region logic
    // ---------------------------
    function getEllipses() {
        const { w, h } = getCanvasCssSize();

        const base = Math.min(w, h) * 0.27;

        const rx = base * 1.15;
        const ry = base * 0.85;

        const cx = w * 0.5;
        const cy = h * 0.47;

        return [
            { key: LISTS[0].key, bit: 1, x: cx - rx * 0.75, y: cy - ry * 0.10, rx, ry },
            { key: LISTS[1].key, bit: 2, x: cx + rx * 0.75, y: cy - ry * 0.10, rx, ry },
            { key: LISTS[2].key, bit: 4, x: cx, y: cy + ry * 0.80, rx, ry },
        ];
    }

    function pointInEllipse(px, py, e) {
        const dx = (px - e.x) / e.rx;
        const dy = (py - e.y) / e.ry;
        return (dx * dx + dy * dy) <= 1;
    }

    function pointInRegion(px, py, ellipses, mask) {
        const inA = pointInEllipse(px, py, ellipses[0]);
        const inB = pointInEllipse(px, py, ellipses[1]);
        const inC = pointInEllipse(px, py, ellipses[2]);

        const wantA = (mask & 1) !== 0;
        const wantB = (mask & 2) !== 0;
        const wantC = (mask & 4) !== 0;

        return (inA === wantA) && (inB === wantB) && (inC === wantC);
    }

    // ---------------------------
    // Drawing (white background, no grid)
    // ---------------------------
    function drawBackground() {
        const { w, h } = getCanvasCssSize();
        ctx.clearRect(0, 0, w, h);

        ctx.save();
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, w, h);
        ctx.restore();
    }

    function drawVenn(ellipses) {
        const { h } = getCanvasCssSize();

        // fills
        for (let i = 0; i < ellipses.length; i++) {
            ctx.beginPath();
            ctx.ellipse(ellipses[i].x, ellipses[i].y, ellipses[i].rx, ellipses[i].ry, 0, 0, Math.PI * 2);
            ctx.fillStyle = LISTS[i].fill;
            ctx.fill();
        }

        // strokes + labels
        ctx.font = "600 18px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial";
        ctx.fillStyle = "rgba(17,24,39,0.90)";
        ctx.textAlign = "center";

        for (let i = 0; i < ellipses.length; i++) {
            ctx.beginPath();
            ctx.ellipse(ellipses[i].x, ellipses[i].y, ellipses[i].rx, ellipses[i].ry, 0, 0, Math.PI * 2);
            ctx.strokeStyle = LISTS[i].stroke;
            ctx.lineWidth = 2.5;
            ctx.stroke();

            const label = LISTS[i].key;

            if (label === "The Guardian") {
                ctx.textBaseline = "top";
                const desiredY = ellipses[i].y + ellipses[i].ry + 10;
                const y = Math.min(desiredY, h - 26);
                ctx.fillText(label, ellipses[i].x, y);
            } else {
                ctx.textBaseline = "bottom";
                const desiredY = ellipses[i].y - ellipses[i].ry - 14;
                const y = Math.max(desiredY, 22);
                ctx.fillText(label, ellipses[i].x, y);
            }
        }
    }

    function computeFontSizes(words) {
        const totals = words.map((w) => w.total);
        const minT = Math.min(...totals);
        const maxT = Math.max(...totals);

        for (const w of words) {
            const t = (maxT === minT) ? 0.5 : (w.total - minT) / (maxT - minT);
            w.fontSize = Math.round(12 + t * 32);
            w.fontWeight = 600;
        }
    }

    function rectsOverlap(a, b) {
        return !(
            a.x + a.w < b.x ||
            a.x > b.x + b.w ||
            a.y + a.h < b.y ||
            a.y > b.y + b.h
        );
    }

    function pickRegionAnchors(ellipses) {
        return new Map([
            [1, { x: ellipses[0].x - ellipses[0].rx * 0.35, y: ellipses[0].y }],
            [2, { x: ellipses[1].x + ellipses[1].rx * 0.35, y: ellipses[1].y }],
            [4, { x: ellipses[2].x, y: ellipses[2].y + ellipses[2].ry * 0.30 }],
            [3, { x: (ellipses[0].x + ellipses[1].x) / 2, y: ellipses[0].y - ellipses[0].ry * 0.05 }],
            [5, { x: (ellipses[0].x + ellipses[2].x) / 2 - ellipses[0].rx * 0.05, y: (ellipses[0].y + ellipses[2].y) / 2 }],
            [6, { x: (ellipses[1].x + ellipses[2].x) / 2 + ellipses[1].rx * 0.05, y: (ellipses[1].y + ellipses[2].y) / 2 }],
            [7, { x: (ellipses[0].x + ellipses[1].x + ellipses[2].x) / 3, y: (ellipses[0].y + ellipses[1].y + ellipses[2].y) / 3 }],
        ]);
    }

    function placeWords(words, ellipses) {
        const placed = [];
        const regionAnchors = pickRegionAnchors(ellipses);
        const sorted = [...words].sort((a, b) => b.fontSize - a.fontSize);

        const { w, h } = getCanvasCssSize();

        for (const wordObj of sorted) {
            const anchor = regionAnchors.get(wordObj.mask) || { x: w / 2, y: h / 2 };

            let found = false;
            const tries = 1600;
            let fontSize = wordObj.fontSize;

            for (let shrink = 0; shrink <= 6 && !found; shrink++) {
                const currentFont = Math.max(MIN_FONT_SIZE, Math.round(fontSize));

                for (let i = 0; i < tries; i++) {
                    const angle = i * 0.45;
                    const radius = 2 + i * 0.085;

                    const px = anchor.x + Math.cos(angle) * radius;
                    const py = anchor.y + Math.sin(angle) * radius;

                    if (px < 18 || px > w - 18 || py < 18 || py > h - 18) continue;
                    if (!pointInRegion(px, py, ellipses, wordObj.mask)) continue;

                    ctx.font = `${wordObj.fontWeight} ${currentFont}px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial`;
                    const metrics = ctx.measureText(wordObj.word);
                    const textW = metrics.width;
                    const textH = currentFont;

                    const rect = {
                        x: px - textW / 2 - 4,
                        y: py - textH / 2 - 3,
                        w: textW + 8,
                        h: textH + 6,
                    };

                    const corners = [
                        { x: rect.x, y: rect.y },
                        { x: rect.x + rect.w, y: rect.y },
                        { x: rect.x, y: rect.y + rect.h },
                        { x: rect.x + rect.w, y: rect.y + rect.h },
                    ];

                    let ok = true;
                    for (const c of corners) {
                        if (!pointInRegion(c.x, c.y, ellipses, wordObj.mask)) { ok = false; break; }
                    }
                    if (!ok) continue;

                    for (const p of placed) {
                        if (rectsOverlap(rect, p.rect)) { ok = false; break; }
                    }
                    if (!ok) continue;

                    placed.push({ ...wordObj, fontSize: currentFont, x: px, y: py, rect });
                    found = true;
                    break;
                }

                fontSize = fontSize * 0.88;
            }
        }

        return placed;
    }

    function drawWords(placed) {
        ctx.save();
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        for (const w of placed) {
            ctx.font = `${w.fontWeight} ${w.fontSize}px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial`;
            const alpha = clamp(0.65 + (w.fontSize - 12) / 70, 0.65, 0.95);
            ctx.fillStyle = `rgba(17,24,39,${alpha})`;

            ctx.shadowColor = "rgba(0,0,0,0.15)";
            ctx.shadowBlur = 3;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 1;

            ctx.fillText(w.word, w.x, w.y);
        }

        ctx.restore();
    }

    // ---------------------------
    // Load + Render Pipeline
    // ---------------------------
    async function loadAndRender() {
        try {
            setStatus("Loading words from database…");
            const rows = await fetchWordsFromApi();

            // Bottom lists (ALL words)
            renderWordListsFromDb(rows);

            // Venn top 50
            const words = buildVennWords(rows);

            if (!words.length) {
                setStatus("No words found in database.");
                return;
            }

            drawBackground();
            const ellipses = getEllipses();
            drawVenn(ellipses);

            computeFontSizes(words);
            const placed = placeWords(words, ellipses);
            drawWords(placed);

            setStatus(`Rendered ${placed.length}/${words.length} words (top ${TOP_N_VENN} by frequency).`);
        } catch (err) {
            console.error(err);
            setStatus(`Could not load words from API (${WORDS_API_URL}). Check backend endpoint + CORS.`);
        }
    }

    // ---------------------------
    // Events
    // ---------------------------
    if (reloadBtn) reloadBtn.addEventListener("click", loadAndRender);

    window.addEventListener("resize", () => {
        fitCanvasToContainer();
        loadAndRender();
    });

    // ---------------------------
    // Init
    // ---------------------------
    fitCanvasToContainer();
    loadAndRender();
})();
