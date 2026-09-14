/* ============================================================================
   داشبورد ارائه — رندر و تعامل
   هیچ عددی در این فایل تایپ نمی‌شود؛ همه از window.__PRES__ می‌آید که
   `python dashboard/build_presentation.py` از روی reports/ و doc/ ساخته است.
   ========================================================================== */
(() => {
"use strict";

const D = window.__PRES__;
if (!D) { document.body.innerHTML = "<p style='padding:40px'>داده‌ی داشبورد لود نشد. اول <code>python dashboard/build_presentation.py</code> را اجرا کنید.</p>"; return; }

/* ------------------------------------------------------------- قالب‌بندی عدد */

const FA = "۰۱۲۳۴۵۶۷۸۹";
const fa = s => String(s).replace(/[0-9]/g, d => FA[+d]);

/** عدد با جداکننده‌ی هزارگان فارسی (U+066C) و ارقام فارسی. */
function n(v, dec = 0) {
  if (v == null || !isFinite(v)) return "—";
  const neg = v < 0;
  const s = Math.abs(v).toFixed(dec);
  let [i, f] = s.split(".");
  i = i.replace(/\B(?=(\d{3})+(?!\d))/g, "٬");
  return (neg ? "−" : "") + fa(f ? i + "." + f : i);
}
const pct   = (v, dec = 1) => v == null ? "—" : fa((v * 100).toFixed(dec)) + "٪";
const mrd   = v => n(v / 1e9, 1) + " میلیارد";
/** واحد را با بزرگیِ عدد هماهنگ کن — «۰.۰ میلیارد» برای پله‌ی اول نردبان بی‌معناست. */
const money = v => Math.abs(v) < 1e9 ? n(v / 1e6, 1) + " میلیون" : mrd(v);
const round = (v, d = 4) => v == null ? "—" : fa(v.toFixed(d));

/* --------------------------------------------------------------- ابزار DOM */

const $  = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));

const set = (sel, html) => { const e = $(sel); if (e) e.innerHTML = html; };

/** جدول از سرستون‌ها و ردیف‌ها. rows: آرایه‌ای از {cells:[…], cls?:string} */
function table(head, rows) {
  const th = head.map(c => `<th>${c}</th>`).join("");
  const tb = rows.map(r =>
    `<tr class="${r.cls || ""}">` + r.cells.map(c =>
      typeof c === "object" ? `<td class="${c.cls || ""}">${c.v}</td>` : `<td>${c}</td>`
    ).join("") + "</tr>").join("");
  return `<table><thead><tr>${th}</tr></thead><tbody>${tb}</tbody></table>`;
}

/** نوارهای افقی ساده. items: {label, value, text, tone, max} */
function bars(items, opts = {}) {
  const max = opts.max ?? Math.max(...items.map(i => i.value));
  return `<div class="bars">` + items.map(i => `
    <div class="bar-row">
      <span class="bl" title="${i.label}">${i.label}</span>
      <span class="bar-track"><span class="bar-fill ${i.tone || ""}" style="width:${(i.value / max * 100).toFixed(2)}%"></span></span>
      <span class="bt">${i.text}</span>
    </div>`).join("") + `</div>`;
}

/* ------------------------------------------------------------------ نمودار */

/** نمودار خطی SVG با محور افقی برگردانده‌شده برای راست‌به‌چپ. */
function lineChart(cfg) {
  const W = 100, H = cfg.h || 58;                 // مختصات نسبی، viewBox
  const pad = { t: 5, r: 9, b: 10, l: 4 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  // ⚠️ راست‌به‌چپ در SVG خودکار نیست — محور X دستی برگردانده می‌شود:
  //    کمینه‌ی X سمت راست می‌نشیند، بیشینه سمت چپ.
  const X = v => W - pad.r - (v - cfg.xMin) / (cfg.xMax - cfg.xMin) * iw;
  const Y = v => pad.t + ih - (v - cfg.yMin) / (cfg.yMax - cfg.yMin) * ih;

  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${cfg.aria || ""}">`;
  for (const g of cfg.yTicks) {
    s += `<line class="grid-l" x1="${pad.l}" x2="${W - pad.r}" y1="${Y(g)}" y2="${Y(g)}"/>`;
    s += `<text class="lbl-s" x="${W - pad.r + 1.5}" y="${Y(g) + 1}" font-size="2.6">${cfg.yFmt(g)}</text>`;
  }
  for (const t of cfg.xTicks)
    s += `<text class="lbl-s" x="${X(t)}" y="${H - 3.4}" font-size="2.6" text-anchor="middle">${cfg.xFmt(t)}</text>`;

  for (const ser of cfg.series) {
    const d = ser.points.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(2)},${Y(p.y).toFixed(2)}`).join(" ");
    if (ser.area)
      s += `<path d="${d} L${X(ser.points.at(-1).x).toFixed(2)},${Y(cfg.yMin)} L${X(ser.points[0].x).toFixed(2)},${Y(cfg.yMin)} Z" fill="${ser.color}" opacity=".1"/>`;
    s += `<path d="${d}" fill="none" stroke="${ser.color}" stroke-width="${ser.w || 1.1}" stroke-linejoin="round" stroke-linecap="round" ${ser.dash ? `stroke-dasharray="${ser.dash}"` : ""}/>`;
    for (const p of ser.points)
      s += `<circle cx="${X(p.x).toFixed(2)}" cy="${Y(p.y).toFixed(2)}" r="${ser.r || .95}" fill="${ser.color}"/>`;
  }
  if (cfg.markerId)
    s += `<line id="${cfg.markerId}" x1="0" x2="0" y1="${pad.t}" y2="${pad.t + ih}" stroke="var(--ink-3)" stroke-width=".5" stroke-dasharray="1.6 1.4"/>`;
  s += `</svg>`;
  s += `<div style="display:none" data-x-min="${cfg.xMin}" data-x-max="${cfg.xMax}" data-pad-r="${pad.r}" data-iw="${iw}" data-w="${W}"></div>`;
  return s;
}

/** نوارهای واگرا حول خط پایه (صفر در وسط). */
function diverging(items, fmt) {
  const m = Math.max(...items.map(i => Math.abs(i.value)));
  return `<div class="dv">` + items.map(i => {
    const w = Math.abs(i.value) / m * 50;
    // محور عددی در فارسی از راست به چپ زیاد می‌شود، مثل محور افقی نمودارهای این صفحه:
    // مقدار منفی (بهتر از خط پایه) سمت راستِ خط وسط، مقدار مثبت سمت چپ.
    const style = i.value < 0 ? `left:50%;width:${w}%` : `right:50%;width:${w}%`;
    return `<div class="dv-row${i.on ? " on" : ""}">
      <span class="dv-l" title="${i.label}">${i.label}</span>
      <span class="dv-track"><span class="dv-zero"></span>
        <span class="dv-fill ${i.value < 0 ? "good" : "bad"}${i.on ? " sig" : ""}" style="${style}"></span></span>
      <span class="dv-v">${fmt(i.value)}</span>
    </div>`;
  }).join("") + `</div>`;
}

/* ═══════════════════════════════════════════════════════ بخش‌های داشبورد ═══ */

const H = D.headline, M = D.meta;

/* ---------------------------------------------------------------- ۰ · قلاب */

function renderHero() {
  set("#h-waste-day", n(H.wastePerDay));
  set("#h-basis", `مبنای همه‌ی اعداد این صفحه: <b>${M.basisNote}</b>`);

  const on = Math.round(H.rhoWeighted * 100);
  $("#waffle").innerHTML = Array.from({ length: 100 },
    (_, i) => `<i class="${i < on ? "on" : ""}"></i>`).join("");
  set("#waffle-cap",
    `از هر <b>۱۰۰</b> پرس غذای رزروشده، <b>${fa(on)}</b> پرس هرگز تحویل گرفته نمی‌شود.<br>
     <span style="opacity:.8">نرخ وزنی حجم: ${pct(H.rhoWeighted, 2)}</span>`);

  const k = [
    { c: "c-short", l: "نرخ عدم‌دریافت", v: pct(H.rhoWeighted, 2),
      s: `میانگین ساده ${pct(H.rhoSimple, 2)} است — عدد وزنی مبناست`, src: H.rhoSrc },
    { c: "c-amber", l: "هدررفت سالانه", v: mrd(H.annualWasteToman), u: "تومان",
      s: `${n(H.annualWastePortions)} پرس در ${fa(M.basis.weeks)} هفته‌ی آموزشی`, src: H.annualWasteSrc },
    { c: "c-save", l: "بازیافتنی با مدل", v: mrd(H.savingsToman), u: "تومان",
      s: `بازه‌ی اطمینان ۹۵٪: ${n(H.savingsTomanLo / 1e9, 1)} تا ${n(H.savingsTomanHi / 1e9, 1)} میلیارد`, src: H.savingsSrc },
    { c: "c-ink", l: "معاوضه در نقطه‌ی عملیاتی", v: n(H.tradeRatio, 1), u: "به ۱",
      s: `${n(H.savedPerDay)} پرس نجات در برابر ${n(H.shortagePerDay)} پرس کمبود، در هر روز سرو`, src: H.tradeSrc },
  ];
  $("#kpis").innerHTML = k.map(x => `
    <div class="kpi ${x.c}">
      <div class="k-label">${x.l}</div>
      <div class="k-value num">${x.v}${x.u ? `<small>${x.u}</small>` : ""}</div>
      <div class="k-sub">${x.s}</div>
      <div class="src">${x.src}</div>
    </div>`).join("");
}

/* -------------------------------------------------------------- ۱ · مسئله */

function renderProblem() {
  const rows = D.tauStar.rows
    .filter(r => r.tau >= 0.02 && r.tau <= 0.5)
    .map(r => ({
      cls: Math.abs(r.tau - M.operatingTau) < 1e-9 ? "hi" : "",
      cells: [
        { v: `<bdi>${n(r.ratio, r.ratio % 1 ? 1 : 0)}×</bdi>`, cls: "n" },
        { v: round(r.tau, 3), cls: "n" },
        { v: Math.abs(r.tau - M.operatingTau) < 1e-9 ? "توصیه‌ی پروژه" : "", cls: "" },
      ],
    }));
  set("#taustar-tbl", table(["نسبت <bdi>C<sub>u</sub>/C<sub>o</sub></bdi>", "چندک هدف", ""], rows));
  set("#taustar-src", D.tauStar.src + " · " + D.tauStar.note);
}

/* ---------------------------------------------------------------- ۲ · داده */

function renderData() {
  $("#scale-tiles").innerHTML = D.scale.map(s => `
    <div class="card flat">
      <div class="k-label">${s.label}</div>
      <div class="k-value num" style="font-size:30px">${n(s.value)}</div>
      <div class="k-sub">${s.note}</div>
    </div>`).join("");

  const cov = new Set(D.coverage.covered);
  set("#coverage-bar", `<div class="cal-strip">` + D.coverage.months.map(m =>
    `<span class="cal-m${cov.has(m) ? " on" : ""}"><b>${m}</b></span>`).join("") + `</div>
    <div class="legend">
      <span><i style="background:var(--save)"></i> پوشش داده (${fa(D.coverage.covered.length)} ماه)</span>
      <span><i style="background:var(--sunk);border:1px solid var(--line)"></i> بدون داده</span>
    </div>`);
}

/* ------------------------------------------------------------- ۳ · یافته‌ها */

function renderInsights() {
  /* سازوکار غیبت و تجزیه‌ی واریانس، هر دو با شکل خودِ ارائه نمایش داده می‌شوند
     (جایگاه‌های insight-mech و insight-var)؛ اینجا فقط عدد و شاهدشان نوشته می‌شود. */
  set("#mech-ratio", n(D.mechanism.ratio, 2));
  set("#mech-src", factSrc(D.mechanism.factId));
  set("#variance-src", factSrc(D.variance.factId));

  /* دو فهرست بدترین */
  const W = D.worst;
  set("#worst-lists", `
    <div class="venn">
      <div class="venn-side"><b class="num">${fa(W.listSize)}</b><span>بدترین به نرخ</span></div>
      <div class="venn-mid"><b class="num">${fa(W.shared)}</b><span>مشترک</span></div>
      <div class="venn-side"><b class="num">${fa(W.listSize)}</b><span>بدترین به پرس</span></div>
    </div>
    ${bars(W.flags.map(f => ({
      label: f.label,
      value: Math.max(f.rate ?? 0, f.abs ?? 0),
      text: `${f.rate == null ? "—" : pct(f.rate, 0)} / ${pct(f.abs, 0)}`,
      tone: (f.rate ?? 0) > (f.abs ?? 0) ? "short" : "ink",
    })), { max: 1 })}
    <div class="legend"><span>سهم هر پرچم تقویمی: <b>فهرست نرخ</b> / <b>فهرست پرس</b></span></div>
    <div class="card flat" style="margin-block-start:12px;font-size:13.5px;line-height:1.85">
      میانه‌ی صدک حجم: فهرست نرخ <b class="num">${fa(W.volumePercentile.rate)}</b> در برابر
      فهرست پرس <b class="num">${fa(W.volumePercentile.abs)}</b> ·
      همبستگی اسپیرمن حجم با نرخ <b class="num">${n(W.spearmanVolumeRate, 3)}</b>،
      حجم با هدررفت مطلق <b class="num">${n(W.spearmanVolumeWaste, 3)}</b>
    </div>`);
  set("#worst-src", factSrc(W.factId) + " · " + factSrc("F81"));

  /* هدف‌گیری */
  const T = D.targeting;
  const tone = { ceiling: "ghost", good: "ink", mid: "mute", base: "amber", bad: "short" };
  set("#targeting-bars", bars(T.rows.map(r => ({
    label: r.label, value: r.value, text: pct(r.value, 2), tone: tone[r.tone],
  })), { max: Math.max(...T.rows.map(r => r.value)) }) +
    `<div class="legend"><span>${T.note}</span></div>`);
  set("#targeting-src", factSrc(T.factId));

  /* تقویم */
  set("#calendar-bars", `<div class="bars">` + D.calendar.rows.map(r => {
    const max = Math.max(...D.calendar.rows.map(x => x.hi ?? x.value));
    const w = r.value / max * 100, lo = r.lo != null ? r.lo / max * 100 : null, hi = r.hi != null ? r.hi / max * 100 : null;
    return `<div class="bar-row">
      <span class="bl" title="${r.label}">${r.label}</span>
      <span class="bar-track" style="position:relative">
        <span class="bar-fill amber" style="width:${w.toFixed(2)}%"></span>
        ${lo != null ? `<span class="whisker" style="right:${lo.toFixed(2)}%;width:${(hi - lo).toFixed(2)}%"></span>` : ""}
      </span>
      <span class="bt">+${round(r.value, 4)}</span>
    </div>`;
  }).join("") + `</div>
    <div class="legend"><span><i style="background:var(--amber)"></i> اندازه‌ی اثر</span>
    <span><i class="wk"></i> بازه‌ی اطمینان ۹۵٪</span></div>`);
  set("#calendar-src", D.calendar.rows.map(r => factSrc(r.factId)).join(" · "));

  /* همه‌ی حقایق */
  $("#facts-list").innerHTML = D.facts.map(f => `
    <div class="fact">
      <span class="fid">${f.id}</span>
      <div class="ftext">${f.fact}</div>
      <div class="fev">${f.evidence}${f.effect ? ` · <b>اندازه اثر:</b> ${f.effect}` : ""}</div>
      <div class="fimp">${f.impact}</div>
    </div>`).join("");
}

const factSrc = id => `doc/data_facts_register.md · ${id}`;

/* ---------------------------------------------------------------- ۴ · مدل */

function renderModel() {
  const maxF = Math.max(...D.funnel.map(f => f.value));
  $("#funnel").innerHTML = D.funnel.map(f => {
    const w = Math.max(14, Math.sqrt(f.value / maxF) * 100);
    return `<div class="fn-row">
      <div class="fn-bar" style="width:${w.toFixed(1)}%">${f.label}</div>
      <div class="fn-n">${f.approx ? `<bdi>~${n(f.value)}</bdi>` : n(f.value)}</div>
    </div>`;
  }).join("");
  set("#funnel-src", D.funnel.map(f => f.src).join(" · "));

  set("#cuts-val", `${fa(D.models.cutsWon)} از ${fa(D.models.cutsTotal)}`);
  set("#cuts-src", D.models.cutsSrc);

  /* نوار واگرا حول خط پایه‌ی B3 */
  set("#model-chart",
    diverging(D.models.rows.map(m => ({
      label: m.id, value: m.delta, on: m.wins,
    })), v => (v < 0 ? "−" : "+") + fa(Math.abs(v).toFixed(5))) +
    `<div class="legend">
      <span><i style="background:var(--save)"></i> بهتر از خط پایه، معنادار</span>
      <span><i style="background:color-mix(in srgb, var(--save) 35%, transparent)"></i> بهتر، ولی نه معنادار</span>
      <span><i style="background:var(--line-2)"></i> بدتر از خط پایه</span>
      <span>خط وسط = خط پایه‌ی <bdi>B3</bdi></span>
    </div>`);

  set("#model-tbl", table(
    ["مدل", "خانواده", "<bdi>pinball</bdi> نرخ", "Δ نسبت به <bdi>B3</bdi>", "بازه‌ی اطمینان ۹۵٪",
     "برد معنادار", "کاهش هدررفت", "نرخ کمبود", "ساعت-هسته", "اجرا"],
    D.models.rows.map(m => ({
      cls: m.wins ? "hi" : "",
      cells: [
        `<bdi>${m.id}</bdi>`, `<bdi>${m.family}</bdi>`,
        { v: round(m.pinball, 5), cls: "n" },
        { v: (m.delta < 0 ? "−" : "+") + fa(Math.abs(m.delta).toFixed(5)), cls: "n" },
        { v: m.ciLo != null ? `<bdi>[${n(m.ciLo, 5)}, ${n(m.ciHi, 5)}]</bdi>`
                            : `<bdi>DM</bdi>`, cls: "n" },
        m.wins ? `<span class="pill win">بله</span>` : `<span class="pill no">—</span>`,
        { v: pct(m.wasteReduction, 1), cls: "n" },
        { v: pct(m.shortageRate, 1), cls: "n" },
        { v: n(m.coreHours, 3), cls: "n" },
        m.compute === "local" ? "محلی" : "کولب/کگل",
      ],
    }))));
  set("#model-src", D.models.src + " — ستون‌های کاهش هدررفت و نرخ کمبود برای اجراهای GPU روی پایه‌ی متفاوتی حساب شده‌اند و با اجراهای محلی مستقیماً قابل‌قیاس نیستند.");

  /* اهمیت فیچر */
  set("#feat-bars", bars(D.features.perm.map(f => ({
    label: `<bdi>${f.name}</bdi>`, value: Math.max(0, f.perm),
    text: fa(f.perm.toFixed(5)), tone: f.perm > 0 ? "ink" : "mute",
  })), { max: Math.max(...D.features.perm.map(f => f.perm)) }));
  set("#feat-src", D.features.src);

  /* نتایج منفی */
  $("#negatives").innerHTML = D.findings
    .map((f, i) => `<li><span class="ix">${fa(i + 1)}</span>
      <div>${f.text}<br><span style="font-size:12.5px;color:var(--ink-3)">
      وضعیت: ${f.status} · شاهد: <bdi>${f.evidence}</bdi></span></div></li>`).join("");
}

/* ------------------------------------------------------------- ۵ · ارزیابی */

function renderEval() {
  set("#holdout-tbl", table(
    ["مدل", "<bdi>pinball</bdi> نرخ", "<bdi>RMSE</bdi>", "<bdi>MAE</bdi>", "<bdi>R²</bdi>",
     "پوشش", "شکاف پوشش", "نرخ کمبود", "کاهش هدررفت", "برد معنادار بر <bdi>B3</bdi>"],
    D.holdout.rows.map(r => ({
      cls: r.id === M.champion ? "hi" : "",
      cells: [
        `<bdi>${r.id}</bdi>`,
        { v: round(r.pinball, 5), cls: "n" }, { v: round(r.rmse, 4), cls: "n" },
        { v: round(r.mae, 4), cls: "n" }, { v: round(r.r2, 3), cls: "n" },
        { v: pct(r.coverage, 1), cls: "n" },
        { v: (r.coverageGap > 0 ? "+" : "") + pct(r.coverageGap, 1), cls: "n" },
        { v: pct(r.shortageRate, 1), cls: "n" }, { v: pct(r.wasteReduction, 1), cls: "n" },
        r.dmP == null ? "—" : (r.dmP < .05 && r.delta < 0
          ? `<span class="pill win">بله · <bdi>p=${n(r.dmP, 4)}</bdi></span>`
          : `<span class="pill no">خیر · <bdi>p=${n(r.dmP, 4)}</bdi></span>`),
      ],
    }))));
  set("#holdout-src", D.holdout.src);

  /* نمودار کالیبراسیون */
  const C = D.calibration.rows;
  set("#calib-chart", lineChart({
    h: 60, xMin: 0, xMax: 1, yMin: 0, yMax: 1,
    xTicks: [0, .25, .5, .75, 1], yTicks: [0, .25, .5, .75, 1],
    xFmt: v => fa(v.toFixed(2)), yFmt: v => fa(v.toFixed(2)),
    aria: "پوشش تجربی در برابر چندک اسمی",
    series: [
      { points: [{ x: 0, y: 0 }, { x: 1, y: 1 }], color: "var(--ink-3)", w: .6, dash: "1.6 1.4", r: 0 },
      { points: C.map(r => ({ x: r.tau, y: r.coverage })), color: "var(--indigo)", w: 1.3 },
    ],
  }) + `<div class="legend">
    <span><i style="background:var(--indigo)"></i> پوشش تجربی</span>
    <span><i style="background:var(--ink-3)"></i> کالیبراسیون کامل (قطر)</span>
    <span>محور افقی: چندک اسمی</span></div>`);

  const A = D.calibration.aci;
  set("#aci-before", (A.before.gap > 0 ? "+" : "") + pct(A.before.gap, 1));
  set("#aci-after",  (A.after.gap  > 0 ? "+" : "") + pct(A.after.gap, 1));
  set("#calib-src", `${D.calibration.src} · ${A.src}`);

  /* برش‌ها */
  const q = D.slices.byQuartile;
  const maxP = Math.max(...q.map(r => r.pinball));
  set("#slice-bars",
    `<div class="card-note" style="font-weight:600;margin-block-end:6px">خطا به تفکیک چارک حجم رزرو</div>` +
    bars(q.map(r => ({ label: r.key, value: r.pinball, text: fa(r.pinball.toFixed(5)), tone: "ink" })), { max: maxP }) +
    `<div class="card-note" style="font-weight:600;margin:16px 0 6px">نرخ کمبود به تفکیک همان چارک‌ها</div>` +
    bars(q.map(r => ({ label: r.key, value: r.shortageRate, text: pct(r.shortageRate, 1), tone: "short" })),
      { max: Math.max(...q.map(r => r.shortageRate)) }));
  set("#slice-src", "reports/phase8/8.3_by_res_quartile.csv · reports/phase8/8.3_by_Meal.csv");
}

/* ---------------------------------------------------------------- ۶ · ارزش */

const TAU = D.tau.points.slice().sort((a, b) => a.tau - b.tau);
const measured = new Set(TAU.map(p => p.tau.toFixed(4)));

/** درون‌یابی خطی بین نقاط واقعی منحنی. */
function atTau(t) {
  const lo = [...TAU].reverse().find(p => p.tau <= t) || TAU[0];
  const hi = TAU.find(p => p.tau >= t) || TAU.at(-1);
  if (lo === hi) return { ...lo, exact: true };
  const w = (t - lo.tau) / (hi.tau - lo.tau);
  const mix = k => lo[k] + (hi[k] - lo[k]) * w;
  return {
    tau: t, exact: false,
    shortageRate: mix("shortageRate"), wasteReduction: mix("wasteReduction"),
    unmetDemand: mix("unmetDemand"), annualPortions: mix("annualPortions"),
    annualToman: mix("annualToman"),
  };
}

// نسبت‌های هزینه‌ای که اسلایدر روی آن‌ها می‌ایستد: هر هفت چندکِ *واقعاً اندازه‌گیری‌شده*
// (تا هیچ نقطه‌ی واقعی از دسترس خارج نماند) به‌علاوه‌ی نسبت‌های گردِ جدول ۱۰.۲ که داخل
// همان بازه می‌افتند. نزولی مرتب می‌شود تا جهتش با محور افقی نمودار یکی باشد:
// زیر dir=rtl کمینه‌ی اسلایدر سمت راست می‌نشیند، و سمت راست نمودار هم چندک کوچک است.
const RATIOS = TAU.map(p => (1 - p.tau) / p.tau)
  .concat(D.tauStar.rows.map(r => r.ratio))
  .filter(r => { const t = 1 / (1 + r); return t >= TAU[0].tau - 1e-6 && t <= TAU.at(-1).tau + 1e-6; })
  .filter((v, i, a) => a.findIndex(x => Math.abs(x - v) < 1e-3) === i)
  .sort((a, b) => b - a);

function renderValue() {
  set("#tau-chart", lineChart({
    h: 62, xMin: TAU[0].tau, xMax: TAU.at(-1).tau, yMin: 0, yMax: .8,
    xTicks: TAU.map(p => p.tau), yTicks: [0, .2, .4, .6, .8],
    xFmt: v => fa(v.toFixed(2)), yFmt: v => pct(v, 0),
    aria: "منحنی معاوضه‌ی کاهش هدررفت در برابر نرخ کمبود",
    markerId: "tau-marker",
    series: [
      { points: TAU.map(p => ({ x: p.tau, y: p.wasteReduction })), color: "var(--save)", w: 1.4, area: true },
      { points: TAU.map(p => ({ x: p.tau, y: p.shortageRate })), color: "var(--short)", w: 1.4 },
      { points: TAU.map(p => ({ x: p.tau, y: p.unmetDemand })), color: "var(--amber)", w: 1.2 },
    ],
  }) + `<div class="card-note" style="text-align:center;font-size:12.5px;margin-block-start:6px">
      محور افقی: چندک هدف — هرچه به چپ، جسورانه‌تر</div>`);
  set("#tau-src", D.tau.src);

  const sl = $("#ratio");
  sl.min = 0; sl.max = RATIOS.length - 1; sl.step = 1;
  sl.value = Math.max(0, RATIOS.findIndex(r => Math.abs(1 / (1 + r) - M.operatingTau) < 1e-6));
  $("#ratio-ticks").innerHTML =
    `<span>${n(RATIOS[0], 0)}× محافظه‌کار</span><span>جسورانه ${n(RATIOS.at(-1), 1)}×</span>`;
  sl.addEventListener("input", updateSim);
  updateSim();

  /* سناریوها */
  set("#scen-tbl", table(
    ["سناریو", "چندک", "نرخ کمبود (رخداد)", "کمبود در هر وعده", "بدترین وعده",
     "کسر تقاضای برآورده‌نشده", "صرفه‌جویی سالانه", "بازه‌ی اطمینان ۹۵٪"],
    D.scenarios.rows.map(s => ({
      cls: s.tau === M.operatingTau ? "hi" : "",
      cells: [
        `<b>${s.name}</b>`, { v: round(s.tau, 2), cls: "n" },
        { v: pct(s.shortageRate, 1), cls: "n" },
        { v: n(s.shortagePerMeal, 1) + " پرس", cls: "n" },
        { v: n(s.worstMeal) + " پرس", cls: "n" },
        { v: pct(s.unmetDemandRate, 2), cls: "n" },
        { v: mrd(s.annualToman) + " تومان", cls: "n" },
        { v: `<bdi>[${n(s.annualTomanLo / 1e9, 1)} – ${n(s.annualTomanHi / 1e9, 1)}]</bdi>`, cls: "n" },
      ],
    }))));
  set("#scen-src", D.scenarios.src);

  /* نردبان */
  const L = D.ladder, maxL = Math.max(...L.map(x => x.portions));
  $("#ladder").innerHTML = L.map(x => `
    <div class="ld-row">
      <div class="ld-l"><b>${x.label}</b><span>${x.note}</span></div>
      <div class="ld-track">
        <div class="ld-fill" style="width:${(Math.sqrt(x.portions / maxL) * 100).toFixed(1)}%"></div>
        ${x.ci ? `<div class="ld-ci" style="right:${(Math.sqrt(x.ci[0] / maxL) * 100).toFixed(1)}%;width:${(Math.sqrt(x.ci[1] / maxL) * 100 - Math.sqrt(x.ci[0] / maxL) * 100).toFixed(1)}%"></div>` : ""}
      </div>
      <div class="ld-v"><b class="num">${n(x.portions, x.portions < 100 ? 1 : 0)}</b> پرس<span class="num">${money(x.portions * M.basis.unitPriceToman)} تومان</span></div>
    </div>`).join("");
  set("#ladder-ratio", n(H.tradeRatio, 1));
  set("#ladder-src", `${D.scenarios.src} — طول میله‌ها ریشه‌ی دوم مقیاس است تا پله‌ی اول دیده شود؛ بازه‌ی اطمینان فقط روی پله‌ی آخر معنا دارد.`);
}

function updateSim() {
  const ratio = RATIOS[+$("#ratio").value];
  const tau = 1 / (1 + ratio);
  const p = atTau(tau);
  const rr = Math.round(ratio * 100) / 100;      // ۱۸.۹۹۹… که از ۱/τ می‌آید، «۱۹» نوشته شود
  set("#r-ratio", n(rr, Number.isInteger(rr) ? 0 : 2) + "×");
  set("#r-tau", round(tau, 3));
  set("#r-saved", n(p.annualPortions));
  set("#r-short", n(shortagePerDayAt(tau)));
  set("#r-toman", mrd(p.annualToman));
  set("#r-fill", pct(1 - p.unmetDemand, 2));

  const exact = measured.has(tau.toFixed(4));
  set("#tau-note", exact
    ? `نقطه‌ی چندک <b class="num">${round(tau, 3)}</b> یکی از هفت نقطه‌ی <b>واقعاً اندازه‌گیری‌شده</b>ی منحنی است.`
    : `چندک <b class="num">${round(tau, 3)}</b> بین دو نقطه‌ی اندازه‌گیری‌شده <b>درون‌یابی خطی</b> شده است — منحنی فقط هفت نقطه‌ی واقعی دارد.`);

  const marker = $("#tau-marker");
  if (marker) {
    const meta = marker.closest(".card").querySelector("[data-x-min]");
    const xMin = +meta.dataset.xMin, xMax = +meta.dataset.xMax;
    const x = +meta.dataset.w - +meta.dataset.padR - (tau - xMin) / (xMax - xMin) * +meta.dataset.iw;
    marker.setAttribute("x1", x); marker.setAttribute("x2", x);
  }
}

/** کمبود روزانه در هر چندک = کسر تقاضای برآورده‌نشده × پرس دریافت‌شده در روز.
 *  سه سناریوی منتشرشده همین عدد را می‌دهند، ولی این رابطه هر هفت نقطه‌ی منحنی را
 *  پوشش می‌دهد — نه فقط سه نقطه‌ای که جدول سناریو دارد. */
const shortagePerDayAt = t => atTau(t).unmetDemand * H.receivedPerDay;

/* ------------------------------------------------------------ گالری شکل‌ها */

/** یک قاب شکل: تصویر + عنوان گزاره‌ای + مسیر شاهد. کلیک = بزرگ‌نمایی. */
function figure(f) {
  return `<figure class="fig">
    <button class="fig-img" type="button" data-zoom="${f.url}" data-cap="${f.title}"
            aria-label="بزرگ‌نمایی: ${f.title}">
      <img src="${f.url}" alt="${f.title}" loading="lazy">
    </button>
    <figcaption>
      <b>${f.title}</b>
      <span>${f.claim}</span>
      <span class="src">${f.src}</span>
    </figcaption>
  </figure>`;
}

/** هر <div class="figs" data-slot="…"> را با شکل‌های همان جایگاه پر کن.
 *  شکل‌هایی که نسخه‌ی تعاملی‌شان بالاتر در همان بخش هست، تاشده می‌آیند. */
function renderFigures() {
  for (const box of $$(".figs")) {
    const mine = D.figures.filter(f => f.slot === box.dataset.slot);
    if (!mine.length) { box.remove(); continue; }
    const open = mine.filter(f => !f.fold), folded = mine.filter(f => f.fold);
    let html = "";
    if (open.length)
      html += `<div class="fig-grid${open.length === 1 ? " one" : ""}">${open.map(figure).join("")}</div>`;
    if (folded.length)
      html += `<details class="acc fig-fold"><summary>${
        folded.length === 1 ? "نمودار متناظر در ارائه" : `${fa(folded.length)} نمودار متناظر در ارائه`
      }</summary><div class="acc-body"><div class="fig-grid${
        folded.length === 1 ? " one" : ""}">${folded.map(figure).join("")}</div></div></details>`;
    box.innerHTML = html;
  }

  /* بزرگ‌نمایی */
  const lb = $("#lightbox"), img = lb.querySelector("img"), cap = lb.querySelector(".lb-cap");
  document.addEventListener("click", e => {
    const b = e.target.closest("[data-zoom]");
    if (b) { img.src = b.dataset.zoom; img.alt = b.dataset.cap; cap.textContent = b.dataset.cap; lb.classList.add("on"); return; }
    if (e.target.closest("#lightbox")) lb.classList.remove("on");
  });
  addEventListener("keydown", e => { if (e.key === "Escape") lb.classList.remove("on"); });
}

/* ------------------------------------------------------------ ۶ · اقتصاد */

function renderEconomics() {
  const E = D.economics;

  const k = [
    { c: "c-amber", l: "هدررفت سالانه، امروز", v: mrd(E.wasteToman), u: "تومان",
      s: `${n(E.wastePortions)} پرس · ${n(E.wastePerDay)} پرس در هر روز سرو` },
    { c: "c-save", l: "بازیافتنی با مدل", v: mrd(E.savedToman), u: "تومان",
      s: `${pct(E.wasteReduction, 0)} هدررفت · ${n(E.savedPortions)} پرس` },
    { c: "c-ink", l: "سود خالص پس از قیمت‌گذاری کمبود", v: mrd(E.netToman), u: "تومان",
      s: `با فرض هزینه‌ی کمبود ${fa(E.cuOverCo)} برابر مازاد — مشتق، نه عدد مخزن` },
    { c: "c-short", l: "هدررفت باقی‌مانده", v: mrd(E.residualToman), u: "تومان",
      s: `${n(E.residualPortions)} پرس · عمداً باقی می‌ماند` },
  ];
  $("#econ-kpis").innerHTML = k.map(x => `
    <div class="kpi ${x.c}">
      <div class="k-label">${x.l}</div>
      <div class="k-value num">${x.v}<small>${x.u}</small></div>
      <div class="k-sub">${x.s}</div>
    </div>`).join("");

  /* سه‌گانه‌ی هدررفت — یک میله‌ی انباشته */
  const share = E.savedPortions / E.wastePortions;
  set("#econ-split", `
    <div class="stack">
      <div class="stack-seg save" style="width:${(share * 100).toFixed(1)}%">
        <b>${pct(share, 0)}</b><span>بازیافتنی</span></div>
      <div class="stack-seg short" style="width:${((1 - share) * 100).toFixed(1)}%">
        <b>${pct(1 - share, 0)}</b><span>باقی‌مانده</span></div>
    </div>
    <div class="stack-legend">
      <span><i style="background:var(--save)"></i> ${n(E.savedPortions)} پرس · ${mrd(E.savedToman)} تومان</span>
      <span><i style="background:var(--short)"></i> ${n(E.residualPortions)} پرس · ${mrd(E.residualToman)} تومان</span>
    </div>`);
  set("#econ-residual-pct", pct(1 - share, 0));
  set("#econ-split-src", `${D.scenarios.src} · ${D.tau.src}`);

  /* نرخ مؤثر، قبل و بعد */
  set("#econ-rho", bars([
    { label: "امروز (پخت روی رزرو)", value: E.rhoBefore, text: pct(E.rhoBefore, 2), tone: "short" },
    { label: "با مدل (پخت روی چندک)", value: E.rhoAfter, text: pct(E.rhoAfter, 2), tone: "" },
  ], { max: E.rhoBefore }));
  set("#econ-rho-b", pct(E.rhoBefore, 2));
  set("#econ-rho-a", pct(E.rhoAfter, 2));
  set("#econ-unmet", pct(1 - E.fillRate, 2));
  set("#econ-rho-src", `doc/data_facts_register.md · F01 · ${D.tau.src}`);

  /* اندازه‌ی کمبود، نه فقط رخدادش */
  set("#econ-shortage-tbl", table(["کمیت", "مقدار", "یعنی چه"], [
    { cells: ["نرخ کمبود سلول", { v: pct(E.shortageRate, 1), cls: "n" },
              { v: `<span class="wrap">کسری از سلول‌ها که <em>اصلاً</em> کمبود دیدند</span>` }] },
    { cells: ["عمق میانگین هر کمبود", { v: n(E.meanDepth, 1) + " پرس", cls: "n" },
              { v: `<span class="wrap">وقتی کمبود رخ می‌دهد، معمولاً این‌قدر است</span>` }] },
    { cells: ["کمبود در هر وعده، کل دانشگاه", { v: n(E.shortagePerMeal, 1) + " پرس", cls: "n" },
              { v: `<span class="wrap">میانه‌اش ${n(E.shortageMedianPerMeal, 1)} پرس است — توزیع چوله</span>` }] },
    { cells: ["بدترین وعده‌ی کل دوره", { v: n(E.shortageWorstMeal) + " پرس", cls: "n" },
              { v: `<span class="wrap">بدترین حالت مشاهده‌شده، نه حالت معمول</span>` }] },
    { cls: "hi", cells: ["کسر تقاضای برآورده‌نشده", { v: pct(1 - E.fillRate, 2), cls: "n" },
              { v: `<span class="wrap">سطح خدمت ${pct(E.fillRate, 2)} — عددی که واقعاً به دانشجو می‌رسد</span>` }] },
    { cls: "hi", cells: ["معاوضه", { v: n(E.tradeRatio, 1) + " به ۱", cls: "n" },
              { v: `<span class="wrap">پرس نجات‌یافته به ازای هر یک پرس کمبود</span>` }] },
  ]));
  set("#econ-depth", n(E.meanDepth, 1));
  set("#econ-shortage-src", D.scenarios.src);

  /* سود خالص */
  const netShare = E.netToman / E.savedToman;
  set("#econ-net", bars([
    { label: "ارزش ناخالص صرفه‌جویی", value: E.savedToman, text: mrd(E.savedToman), tone: "" },
    { label: `هزینه‌ی کمبود (${fa(E.cuOverCo)}× مازاد)`, value: E.shortageCostToman,
      text: mrd(E.shortageCostToman), tone: "short" },
    { label: "سود خالص", value: E.netToman, text: mrd(E.netToman), tone: "ink" },
  ], { max: E.savedToman }));
  set("#econ-net-note",
    `فرض: هزینه‌ی هر پرس کمبود <bdi>${fa(E.cuOverCo)}×</bdi> هزینه‌ی هر پرس مازاد ` +
    `(${n(E.unitPriceToman)} تومان) — همان نسبتی که چندک ${round(M.operatingTau, 2)} از آن می‌آید. ` +
    `کمبود سالانه ${n(E.shortageAnnualPortions)} پرس برآورد شده است. ` +
    `این یک محاسبه‌ی مشتق است و در هیچ گزارشی از مخزن منتشر نشده.`);
  set("#econ-net-share", pct(netShare, 0));
  set("#econ-net-src", `${D.scenarios.src} · doc/decision_log.md ردیف ۳۴ (برآورد نسبت هزینه)`);

  /* دوراهی یک وعده */
  const L = D.dilemma, price = E.unitPriceToman;
  set("#econ-dilemma", `
    <div class="duo">
      <div class="duo-side bad">
        <div class="duo-h">اگر به‌اندازه‌ی رزرو بپزیم</div>
        <div class="duo-n num">${n(L.res)}</div>
        <div class="duo-s">پرس پخت برای ${n(L.res)} رزرو</div>
        <div class="duo-r">${fa(L.absent)} نفر نمی‌آیند ⟵
          <b>${fa(L.absent)} پرس هدررفت</b> ≈ ${n(L.absent * price / 1e6, 1)} میلیون تومان</div>
      </div>
      <div class="duo-side warn">
        <div class="duo-h">اگر بی‌ضابطه کم بپزیم</div>
        <div class="duo-n num">${n(L.cookLow)}</div>
        <div class="duo-s">پرس پخت، بدون مدل</div>
        <div class="duo-r">${n(L.demand)} نفر می‌آیند ⟵
          <b>${fa(L.demand - L.cookLow)} پرس کمبود</b> و صف خالی</div>
      </div>
    </div>`);
  set("#econ-dilemma-src", L.src);

  /* یک سلف مشخص */
  const S = D.spotlight;
  set("#spot-name", S.name);
  set("#spot-body", `
    <p class="card-note">${S.note}</p>
    <div class="spot">
      <div><span class="o-l">میانگین رزرو هر وعده</span><b class="num">${n(S.meanRes)}</b></div>
      <div><span class="o-l">نرخ پیش‌بینی‌شده</span><b class="num">${pct(S.rhoHat, 1)}</b></div>
      <div class="hi"><span class="o-l">پخت پیشنهادی</span><b class="num">${n(S.cook)}</b></div>
      <div class="hi"><span class="o-l">صرفه‌جویی هر وعده</span><b class="num">${n(S.savedPortions)} پرس</b></div>
    </div>
    <p class="card-note" style="margin-block-start:12px">
      یعنی <b>${n(S.savedPortions * E.unitPriceToman / 1e6, 1)} میلیون تومان در هر وعده</b>،
      و حدود <b>${mrd(S.savedPortions * E.servingDays * E.unitPriceToman)} تومان در سال</b>
      فقط از همین یک سلف.
    </p>`);
  set("#spot-caveat", S.caveat);
  set("#spot-src", S.src);

  /* چرا محافظه‌کارانه */
  set("#econ-rpd-f", n(E.rowsPerDayFolds, 1));
  set("#econ-rpd-a", n(E.rowsPerDayAll, 1));
  set("#econ-basis", bars([
    { label: "کل داده", value: E.rowsPerDayAll, text: n(E.rowsPerDayAll, 1) + " ردیف/روز", tone: "mute" },
    { label: "مبنای این برآورد", value: E.rowsPerDayFolds, text: n(E.rowsPerDayFolds, 1) + " ردیف/روز", tone: "" },
  ], { max: E.rowsPerDayAll }));
  set("#econ-basis-gap", pct(1 - E.rowsPerDayFolds / E.rowsPerDayAll, 0));
  set("#econ-basis-src", `${D.scenarios.src} · doc/data_manifest.md`);

  /* نمونه‌های واقعی */
  set("#examples-tbl", table(
    ["سلف", "وعده", "رزرو", "پخت پیشنهادی", "صرفه‌جویی", "ارزش", "دریافت واقعی", "نتیجه"],
    D.examples.rows.map(r => ({
      cls: r.gap < 0 ? "" : "hi",
      cells: [
        { v: `<span class="wrap">${r.restaurant}</span>` }, r.meal,
        { v: n(r.res), cls: "n" }, { v: `<b>${n(r.cook)}</b>`, cls: "n" },
        { v: n(r.savedPortions) + " پرس", cls: "n" },
        { v: n(r.savedPortions * E.unitPriceToman / 1e6, 1) + " میلیون", cls: "n" },
        { v: n(r.recv), cls: "n" },
        r.gap < 0
          ? `<span class="pill bad">کمبود ${fa(Math.abs(r.gap))} پرس</span>`
          : `<span class="pill win">بدون کمبود · ${fa(r.gap)} پرس مازاد</span>`,
      ],
    }))));
  set("#examples-src", D.examples.src);
}

/* -------------------------------------------------------------- ۹ · شواهد */

function renderEvidence() {
  const rows = [
    ["سرخط‌ها و برآورد سالانه", "reports/phase10/10.3_policy_scenarios.csv · 10.4_annual_savings.md"],
    ["منحنی معاوضه‌ی چندک", D.tau.src],
    ["نگاشت نسبت هزینه به چندک", D.tauStar.src],
    ["جدول مقایسه‌ی مدل‌ها", D.models.src],
    ["برش‌های اجباری", D.models.cutsSrc],
    ["اهمیت فیچر", D.features.src],
    ["کالیبراسیون", D.calibration.src],
    ["ارزیابی روی پنجره‌ی آزمون", D.holdout.src],
    ["تجزیه‌ی خطا به تفکیک برش", "reports/phase8/8.3_*.csv"],
    ["حقایق داده", "doc/data_facts_register.md"],
    ["یافته‌های مدل‌سازی", "doc/findings_register.md"],
    ["تصمیم‌های روش‌شناختی", "doc/decision_log.md"],
  ];
  set("#sources-tbl", table(["بخش داشبورد", "فایل شاهد"],
    rows.map(r => ({ cells: [{ v: `<span class="wrap">${r[0]}</span>` }, { v: `<span class="path">${r[1]}</span>` }] }))));

  set("#foot-meta",
    `ساخت داده: <bdi>${fa(M.builtAt)}</bdi> · کامیت <bdi>${M.gitCommit}</bdi> · قهرمان: <bdi>${M.champion}</bdi>`);
}

/* ══════════════════════════════════════════════════════════ پوسته و تعامل ═══ */

const SECTIONS = [
  ["hook", "سرآغاز"], ["problem", "مسئله"], ["data", "داده"], ["insight", "یافته‌ها"],
  ["model", "مدل"], ["eval", "ارزیابی"], ["value", "اقتصاد"], ["limits", "محدودیت"],
  ["next", "گام بعد"], ["evidence", "شواهد"],
];

function chrome() {
  $("#nav").innerHTML = SECTIONS.map(([id, t]) => `<a href="#${id}" data-t="${id}">${t}</a>`).join("");

  /* پررنگ‌کردن بخش فعال + نوار پیشرفت */
  const links = new Map($$("#nav a").map(a => [a.dataset.t, a]));
  const io = new IntersectionObserver(es => {
    for (const e of es) if (e.isIntersecting) {
      links.forEach(a => a.classList.remove("on"));
      links.get(e.target.id)?.classList.add("on");
    }
  }, { rootMargin: "-45% 0px -50% 0px" });
  SECTIONS.forEach(([id]) => { const s = document.getElementById(id); if (s) io.observe(s); });

  const prog = $("#progress");
  const onScroll = () => {
    const max = document.body.scrollHeight - innerHeight;
    prog.style.width = (max > 0 ? (scrollY / max) * 100 : 0) + "%";
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* پوسته */
  const root = document.documentElement;
  const saved = (() => { try { return localStorage.getItem("pres-theme"); } catch { return null; } })();
  const apply = t => {
    root.dataset.theme = t;
    $("#themeBtn").textContent = t === "dark" ? "روشن" : "تیره";
  };
  apply(saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  $("#themeBtn").onclick = () => {
    const t = root.dataset.theme === "dark" ? "light" : "dark";
    apply(t);
    try { localStorage.setItem("pres-theme", t); } catch {}
  };

  /* راهروی شاهد */
  $("#srcBtn").onclick = e => {
    const on = document.body.classList.toggle("show-src");
    e.currentTarget.setAttribute("aria-pressed", String(on));
  };

  $("#printBtn").onclick = () => {
    $$("details.acc").forEach(d => d.open = true);
    print();
  };
}

/* ------------------------------------------------------------------- اجرا */

renderHero();
renderProblem();
renderData();
renderInsights();
renderModel();
renderEval();
renderValue();
renderEconomics();
renderFigures();
renderEvidence();
chrome();

})();
