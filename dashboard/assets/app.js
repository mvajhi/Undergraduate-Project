/* داشبورد دانش پروژه — رندر و تعامل. بدون وابستگی بیرونی. */
(function () {
  "use strict";

  var D = window.__DASH__;
  if (!D) { document.body.innerHTML = '<div class="empty-state">فایل داده بارگذاری نشد. اول <code>python dashboard/build.py</code> را اجرا کنید.</div>'; return; }

  /* ------------------------------------------------------------- ابزارها */

  var FA = "۰۱۲۳۴۵۶۷۸۹";
  function fa(s) {
    return String(s)
      .replace(/(\d)\.(\d)/g, "$1\u066B$2")   // ممیز فارسی، نه نقطه‌ی لاتین
      .replace(/[0-9]/g, function (d) { return FA[+d]; });
  }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function num(v, d) { return v == null || isNaN(v) ? "—" : Number(v).toFixed(d == null ? 3 : d); }
  function pct(v, d) { return v == null || isNaN(v) ? "—" : fa((v * 100).toFixed(d == null ? 1 : d)) + "٪"; }
  function pctRaw(v, d) { return v == null || isNaN(v) ? "—" : fa(Number(v).toFixed(d == null ? 1 : d)) + "٪"; }
  function group(n) { return fa(String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, "٬")); }
  function milyard(t) { return t == null ? "—" : fa((t / 1e9).toFixed(1)); }
  function code(s) { return '<bdi class="code">' + esc(s) + "</bdi>"; }
  function srcLink(p) { return p ? '<span class="src" title="مسیر شاهد در مخزن">' + esc(p) + "</span>" : ""; }
  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  // متن ساده برای جست‌وجو (بی‌تفاوت به نیم‌فاصله، ی/ك عربی و رقم فارسی)
  function norm(s) {
    return String(s == null ? "" : s)
      .replace(/[‌‏‎]/g, "")
      .replace(/ي/g, "ی").replace(/ك/g, "ک")
      .replace(/[۰-۹]/g, function (d) { return String(FA.indexOf(d)); })
      .toLowerCase();
  }
  function match(hay, needle) {
    if (!needle) return true;
    var h = norm(hay);
    return norm(needle).split(/\s+/).filter(Boolean).every(function (t) { return h.indexOf(t) !== -1; });
  }

  /* -------------------------------------------------------------- نمودارها */

  var C = { accent: "var(--accent)", ok: "var(--ok)", bad: "var(--bad)", warn: "var(--warn)", info: "var(--info)", muted: "var(--faint)" };

  /**
   * میله‌ی افقی با HTML واقعی (نه SVG): متنِ برچسب هیچ‌وقت مقیاس نمی‌خورد و
   * راست‌به‌چپ بودن را از جریان صفحه می‌گیرد. میله از راست به چپ رشد می‌کند.
   * برای داده‌ی دوطرفه `diverging:true` بدهید تا صفر وسط بنشیند.
   */
  function hbar(rows, opts) {
    opts = opts || {};
    var vals = rows.map(function (r) { return r.value || 0; });
    var max = opts.max != null ? opts.max : Math.max.apply(null, vals.map(Math.abs));
    if (!max) max = 1;
    var out = ['<div class="hbar' + (opts.diverging ? " diverging" : "") + '">'];
    rows.forEach(function (r) {
      var w = Math.max(.8, (Math.abs(r.value) / max) * (opts.diverging ? 50 : 100));
      var color = r.color || opts.color || C.accent;
      var side = opts.diverging && r.value > 0 ? "neg" : "pos";
      out.push(
        '<div class="hbar-row' + (r.dim ? " dim" : "") + '">' +
          '<div class="hbar-label" title="' + esc(r.label) + '">' + (opts.rawLabel ? r.label : esc(r.label)) + "</div>" +
          '<div class="hbar-track"><span class="fill ' + side + '" style="width:' + w.toFixed(2) + "%;background:" + color + '"></span></div>' +
          '<div class="hbar-val">' + esc(r.text != null ? r.text : num(r.value)) + "</div>" +
        "</div>"
      );
    });
    out.push("</div>");
    return out.join("");
  }

  /** نمودار خطی چندسری با محور افقی صعودی (هم‌راستا با شکل‌های matplotlib مخزن). */
  function lineChart(series, opts) {
    opts = opts || {};
    var W = opts.width || 760, H = opts.height || 250;
    var ml = 54, mr = 16, mt = 12, mb = 34;
    var xs = [], ys = [];
    series.forEach(function (s) { s.points.forEach(function (p) { xs.push(p[0]); ys.push(p[1]); }); });
    var x0 = opts.xMin != null ? opts.xMin : Math.min.apply(null, xs), x1 = opts.xMax != null ? opts.xMax : Math.max.apply(null, xs);
    var y0 = opts.yMin != null ? opts.yMin : Math.min.apply(null, ys), y1 = opts.yMax != null ? opts.yMax : Math.max.apply(null, ys);
    if (y1 === y0) y1 = y0 + 1;
    var pad = (y1 - y0) * .08; y0 -= pad; y1 += pad;
    var PX = function (x) { return ml + ((x - x0) / (x1 - x0 || 1)) * (W - ml - mr); };
    var PY = function (y) { return H - mb - ((y - y0) / (y1 - y0)) * (H - mt - mb); };
    var out = ['<div class="chart" dir="ltr"><svg viewBox="0 0 ' + W + " " + H + '" width="100%" height="' + H + '" role="img">'];
    for (var i = 0; i <= 4; i++) {
      var gy = y0 + (i / 4) * (y1 - y0), py = PY(gy);
      out.push('<line class="grid-line" x1="' + ml + '" y1="' + py.toFixed(1) + '" x2="' + (W - mr) + '" y2="' + py.toFixed(1) + '"/>');
      out.push('<text class="val" x="' + (ml - 7) + '" y="' + (py + 3).toFixed(1) + '" text-anchor="end">' + (opts.yFmt ? opts.yFmt(gy) : num(gy, 2)) + "</text>");
    }
    (opts.xTicks || []).forEach(function (t) {
      out.push('<text class="val" x="' + PX(t).toFixed(1) + '" y="' + (H - mb + 17) + '" text-anchor="middle">' + (opts.xFmt ? opts.xFmt(t) : num(t, 2)) + "</text>");
    });
    out.push('<line class="grid-line" x1="' + ml + '" y1="' + (H - mb) + '" x2="' + (W - mr) + '" y2="' + (H - mb) + '" stroke-dasharray="0"/>');
    series.forEach(function (s) {
      var d = s.points.map(function (p, i) { return (i ? "L" : "M") + PX(p[0]).toFixed(1) + " " + PY(p[1]).toFixed(1); }).join(" ");
      out.push('<path d="' + d + '" fill="none" stroke="' + s.color + '" stroke-width="2.2" stroke-linejoin="round"/>');
      s.points.forEach(function (p) {
        out.push('<circle cx="' + PX(p[0]).toFixed(1) + '" cy="' + PY(p[1]).toFixed(1) + '" r="3" fill="' + s.color + '"/>');
      });
    });
    if (opts.markX != null) {
      out.push('<line x1="' + PX(opts.markX).toFixed(1) + '" y1="' + mt + '" x2="' + PX(opts.markX).toFixed(1) + '" y2="' + (H - mb) + '" stroke="var(--text)" stroke-width="1.4" stroke-dasharray="4 3" opacity=".65"/>');
    }
    if (opts.xLabel) out.push('<text x="' + ((ml + W - mr) / 2) + '" y="' + (H - 3) + '" text-anchor="middle" direction="rtl">' + esc(opts.xLabel) + "</text>");
    out.push("</svg></div>");
    var leg = series.map(function (s) { return '<span><i style="background:' + s.color + '"></i>' + esc(s.name) + "</span>"; }).join("");
    return out.join("") + '<div class="legend">' + leg + "</div>";
  }

  /* -------------------------------------------------- جدول مرتب‌شونده‌ی عمومی */

  var sortState = {};
  function sortableTable(id, cols, rows, opts) {
    opts = opts || {};
    var st = sortState[id] || (sortState[id] = { key: opts.sortKey || null, dir: opts.sortDir || 1 });
    if (st.key) {
      var col = cols.filter(function (c) { return c.key === st.key; })[0];
      rows = rows.slice().sort(function (a, b) {
        var va = col.sortVal ? col.sortVal(a) : a[st.key], vb = col.sortVal ? col.sortVal(b) : b[st.key];
        if (va == null) return 1; if (vb == null) return -1;
        if (typeof va === "number" && typeof vb === "number") return (va - vb) * st.dir;
        return String(va).localeCompare(String(vb), "fa") * st.dir;
      });
    }
    var head = cols.map(function (c) {
      var arrow = st.key === c.key ? ' <span class="arrow">' + (st.dir === 1 ? "▲" : "▼") + "</span>" : "";
      return '<th class="' + (c.num ? "num " : "") + 'sortable" data-table="' + id + '" data-key="' + c.key + '">' + esc(c.title) + arrow + "</th>";
    }).join("");
    var body = rows.map(function (r) {
      return "<tr>" + cols.map(function (c) {
        return '<td class="' + (c.num ? "num" : "") + (c.wide ? " wide" : "") + '">' + (c.render ? c.render(r) : esc(r[c.key] == null ? "—" : r[c.key])) + "</td>";
      }).join("") + "</tr>";
    }).join("");
    if (!rows.length) body = '<tr><td colspan="' + cols.length + '"><div class="empty-state">چیزی با این فیلترها پیدا نشد.</div></td></tr>';
    return '<div class="table-wrap"><table class="' + (opts.compact ? "table-compact" : "") + '"><thead><tr>' + head + "</tr></thead><tbody>" + body + "</tbody></table></div>";
  }

  document.addEventListener("click", function (e) {
    var th = e.target.closest && e.target.closest("th.sortable");
    if (!th) return;
    var id = th.dataset.table, key = th.dataset.key, st = sortState[id] || (sortState[id] = {});
    if (st.key === key) st.dir = -st.dir; else { st.key = key; st.dir = key === "id" ? 1 : 1; }
    render();
  });

  /* ------------------------------------------------------------ وضعیت فیلتر */

  var S = {
    tab: "overview",
    factQ: "", factSection: "", factStar: false,
    findQ: "", findStatus: "", findPhase: "",
    decQ: "",
    featQ: "", featFamily: "",
    modelQ: "", modelWin: false,
    figQ: "", figGroup: "",
    docQ: "", docGroup: "",
    phaseOpen: {},
    breakdown: "dow_name",
    tau: 0.2,
  };

  /* ------------------------------------------------------------- پنل: کلیات */

  function panelOverview() {
    var m = D.meta, p = D.project, ch = p.champion || {}, ag = p.aggressive || {};
    var hold = (D.evaluation.holdout || []);
    var sig = (D.models.comparison || []).filter(function (x) { return x.beatsB3; }).length;
    var f = D.facts.length, fi = D.findings.length;

    var kpis = [
      { l: "نرخ عدم‌دریافت پایه (وزنی)", v: "۸٫۰۵٪", f: "F01 · dataset_v2" },
      { l: "مقیاس داده‌ی فردی", v: "۲٬۰۴۹٬۳۲۲", f: "رزرو · ۲۶٬۷۶۸ فرد · ۱۴۲ روز · ۳۰ سلف" },
      { l: "قهرمان فاز ۷", v: "<bdi class=\"code\">" + esc(ch.model || "—") + "</bdi>", sm: true, f: "pinball نرخ " + num(ch.pinball, 5) + " · خانواده " + esc(ch.family || "") },
      { l: "کاهش هدررفت قهرمان", v: pct(ch.wasteReduction, 1), f: "نرخ کمبود " + pct(ch.shortage, 1) + " · τ=۰٫۲۰" },
      { l: "صرفه‌جویی سالانه (τ=۰٫۲۰)", v: milyard(ag.annualToman) + " میلیارد", f: "بازه‌ی اطمینان ۹۵٪: " + milyard(ag.annualTomanLo) + " تا " + milyard(ag.annualTomanHi) + " · مبنای ۳۲ هفته، ۳۷۵ هزار تومان هر پرس" },
      { l: "مدل‌های با برد آماری بر B3", v: fa(sig) + " از " + fa(D.models.comparison.length), f: "بوت‌استرپ بلوکی دوبعدی / DM-test" },
    ];

    var html = '<div class="grid k3">' + kpis.map(function (k) {
      return '<div class="kpi"><div class="label">' + k.l + '</div><div class="value' + (k.sm ? " sm" : "") + '">' + k.v + '</div><div class="foot">' + k.f + "</div></div>";
    }).join("") + "</div>";

    // پیشرفت فازها
    var doneP = D.phases.filter(function (p) { return p.pct === 100; }).length;
    html += '<div class="card"><h3>پیشرفت فازها</h3><div class="card-note">' +
      fa(m.counts.phaseItemsDone) + " از " + fa(m.counts.phaseItems) + " بسته‌ی کاری تیک خورده · " +
      fa(doneP) + " فاز از " + fa(D.phases.length) + " کامل · منبع: <span class=\"src\">doc/progress/</span></div>";
    html += '<div class="grid k4">' + D.phases.map(function (p) {
      return '<div><div style="display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:3px"><span>' +
        '<span class="src">' + fa(p.code) + "</span> " + esc(p.title) + '</span><span class="ratio" style="font-family:var(--mono);color:var(--muted)">' +
        fa(p.done) + "/" + fa(p.total) + '</span></div><div class="bar' + (p.pct === 100 ? " done" : "") + '"><span style="width:' + p.pct + '%"></span></div></div>';
    }).join("") + "</div></div>";

    // Milestone ها
    html += '<div class="grid k2"><div class="card"><h3>Milestone ها</h3><div class="card-note">دروازه‌های پروژه طبق <span class="src">AGENTS.md</span></div>' +
      p.milestones.map(function (ms) {
        var ph = D.phases.filter(function (x) { return x.code === ms.phase; })[0] || { pct: 0 };
        var cls = ph.pct === 100 ? "ok" : ph.pct > 0 ? "warn" : "neutral";
        return '<div style="display:flex;gap:9px;align-items:center;padding:5px 0"><span class="badge id ' + cls + '">' + ms.id +
          '</span><span style="flex:1">' + esc(ms.title) + '</span><span class="src">' + fa(ph.pct) + "٪</span></div>";
      }).join("") + "</div>";

    // شمارش دانش
    var counts = [
      ["حقایق EDA", m.counts.facts, "facts"], ["یافته‌ها", m.counts.findings, "findings"],
      ["تصمیم‌ها", m.counts.decisions, "decisions"], ["فیچرها", m.counts.features, "features"],
      ["مدل‌های مقایسه‌شده", m.counts.models, "models"], ["کارت مدل", m.counts.modelCards, "models"],
      ["شکل", m.counts.figures, "figures"], ["سند markdown", m.counts.docs, "docs"],
    ];
    html += '<div class="card"><h3>دانش تجمیع‌شده در این داشبورد</h3><div class="card-note">هر عدد به بخش خودش لینک است</div><div class="grid k2" style="gap:8px">' +
      counts.map(function (c) {
        return '<a href="#' + c[2] + '" style="display:flex;justify-content:space-between;padding:5px 9px;border-radius:7px;background:var(--surface-2)"><span>' +
          c[0] + '</span><b style="font-family:var(--mono)">' + fa(c[1]) + "</b></a>";
      }).join("") + "</div></div></div>";

    // هشدار کیفیت داده
    var dq = m.dataQuality || {};
    if ((dq.decisionIdCollisions || []).length) {
      html += '<div class="card"><h3>هشدار کیفیت داده‌ی مستندات</h3><div class="note warn"><strong>برخورد شناسه در <span class="src">doc/decision_log.md</span>:</strong> ' +
        'شماره‌ی ' + dq.decisionIdCollisions.map(fa).join("، ") + ' برای بیش از یک تصمیم کاملاً متفاوت استفاده شده است. ' +
        'چون ردیف‌های این دفترچه از جاهای دیگر مخزن با شماره ارجاع داده می‌شوند، داشبورد شماره را عوض نمی‌کند و ' +
        'مثل قاعده‌ی خودِ <span class="src">findings_register.md</span> برای ۲۸/۲۹، پسوند الف/ب می‌گذارد.</div></div>';
    }
    return html;
  }

  /* -------------------------------------------------------------- پنل: فازها */

  function panelPhases() {
    var html = '<div class="grid k2">';
    html += D.phases.map(function (p) {
      var open = !!S.phaseOpen[p.code];
      var items = "";
      if (open) {
        var lastG = null;
        items = '<div class="phase-items">' + p.items.map(function (it) {
          var g = it.group !== lastG ? '<div class="group">' + esc(it.group) + "</div>" : "";
          lastG = it.group;
          return g + '<div class="item ' + (it.done ? "done" : "todo") + '"><span class="mark">' + (it.done ? "✔" : "◻") + "</span><span>" + esc(it.text) + "</span></div>";
        }).join("") + "</div>";
      }
      return '<div class="card phase-card" data-phase="' + p.code + '">' +
        '<div class="head"><h3><span class="num">فاز ' + fa(p.code) + "</span> " + esc(p.title) + '</h3><span class="ratio">' + fa(p.done) + "/" + fa(p.total) + '</span></div>' +
        '<div class="bar' + (p.pct === 100 ? " done" : "") + '" style="margin:7px 0 5px"><span style="width:' + p.pct + '%"></span></div>' +
        '<div class="src">' + esc(p.file) + "</div>" + items + "</div>";
    }).join("");
    return html + "</div>";
  }

  /* ------------------------------------------------------- پنل: داده و قواعد */

  function panelData() {
    var p = D.project;
    var html = '<div class="card"><h3>سطوح داده</h3><div class="card-note">تفکیک «مدل A / مدل B» از فاز ۷ حذف شده؛ جایش را سطح داده به‌عنوان محور آزمایش گرفته است.</div>' +
      '<div class="table-wrap"><table><thead><tr><th>سطح</th><th>واحد رکورد</th><th>حجم</th><th>نقش</th></tr></thead><tbody>' +
      p.levels.map(function (l) {
        return "<tr><td><b>" + l.id + "</b></td><td>" + esc(l.unit) + "</td><td>" + esc(l.size) + "</td><td>" + esc(l.note) + "</td></tr>";
      }).join("") + "</tbody></table></div></div>";

    html += '<div class="card"><h3>قاعده‌ی برش اطلاعاتی</h3><div class="card-note">مهم‌ترین قانون پروژه. لحظه‌ی پیش‌بینی به تفکیک وعده است، نه یکسان برای کل روز.</div>' +
      '<div class="table-wrap"><table><thead><tr><th>وعده‌ی هدف</th><th>لحظه‌ی برش</th><th>آخرین outcome در دسترس</th><th>ممنوع (با اینکه تاریخش ≤ d−۱ است)</th></tr></thead><tbody>' +
      p.cutoff.map(function (c) {
        return "<tr><td>" + esc(c.target) + "</td><td>" + esc(c.cut) + "</td><td>" + esc(c.last) +
          '</td><td><span class="badge bad">' + esc(c.forbidden) + "</span></td></tr>";
      }).join("") + "</tbody></table></div>" +
      '<div class="note warn" style="margin-top:12px"><strong>تله‌ی رایج:</strong> «هر d′ ≤ d−۱ مجاز است» کافی نیست. شام روز d−۱ در ساعت ۱۵ همان روز (لحظه‌ی برش ناهار روز d) هنوز سرو نشده — با اینکه تاریخش d−۱ است، ممنوع است. جزئیات: <span class="src">doc/project-definition.md §۴-۲</span> · ردیف ۹ <span class="src">decision_log</span></div></div>';

    html += '<div class="grid k2"><div class="card"><h3>فصل‌های گزارش نهایی</h3><div class="card-note">نقشه‌ی ادعا ← شاهد در <span class="src">doc/report-source-map.md</span></div>' +
      '<ol style="padding-inline-start:1.4em;margin:0">' + p.chapters.map(function (c) { return "<li>" + esc(c) + "</li>"; }).join("") + "</ol></div>";

    html += '<div class="card"><h3>اسناد پایه‌ی داده</h3><div class="card-note">شناسنامه، دیکشنری و ممیزی نشت</div><dl class="kv">' +
      [["doc/data_manifest.md", "هش snapshot هر فایل خام، قفل cv_folds"],
       ["doc/data_dictionary.md", "دیکشنری داده‌ی تجمیعی"],
       ["doc/data_dictionary_individual.md", "دیکشنری داده‌ی سطح فردی"],
       ["doc/leakage_audit.md", "ممیزی نشت هر فیچر در برابر قاعده‌ی برش"],
       ["doc/data_requests.md", "اقلام داده‌ی درخواستی از ذی‌نفع"]].map(function (r) {
        return "<dt>" + srcLink(r[0]) + "</dt><dd>" + esc(r[1]) + "</dd>";
      }).join("") + "</dl></div></div>";
    return html;
  }

  /* ------------------------------------------------------------- پنل: حقایق */

  // «الف) شکل متغیر هدف» → {letter: "الف", rest: "شکل متغیر هدف"}
  function splitSection(sec) {
    var m = /^\s*([^\s)]{1,3})\)\s*(.*)$/.exec(sec);
    return m ? { letter: m[1], rest: m[2] } : { letter: "", rest: sec };
  }

  function panelFacts() {
    var sections = [];
    D.facts.forEach(function (f) { if (sections.indexOf(f.section) === -1) sections.push(f.section); });
    var rows = D.facts.filter(function (f) {
      if (S.factSection && f.section !== S.factSection) return false;
      if (S.factStar && !f.starred) return false;
      return match([f.id, f.fact, f.evidence, f.effect, f.impact].join(" "), S.factQ);
    });

    var chips = '<div class="chip-row"><input class="filter-input" id="factQ" placeholder="جست‌وجو در حقایق…" value="' + esc(S.factQ) + '">' +
      '<button class="chip' + (S.factSection ? "" : " on") + '" data-fsec="">همه</button>' +
      sections.map(function (s) {
        var p = splitSection(s);
        return '<button class="chip' + (S.factSection === s ? " on" : "") + '" data-fsec="' + esc(s) + '" title="' + esc(s) + '">' +
          (p.letter ? '<b style="opacity:.6">' + esc(p.letter) + "</b> " : "") + esc(p.rest) + "</button>";
      }).join("") +
      '<button class="chip' + (S.factStar ? " on" : "") + '" data-fstar="1">⭐ ستاره‌دار</button>' +
      '<span class="result-count">' + fa(rows.length) + " از " + fa(D.facts.length) + "</span></div>";

    var cols = [
      { key: "id", title: "شناسه", sortVal: function (r) { return +r.id.slice(1); }, render: function (r) { return '<span class="badge id ' + (r.starred ? "warn" : "neutral") + '">' + r.id + "</span>"; } },
      { key: "fact", title: "حقیقت", wide: true, render: function (r) {
          return "<b>" + esc(r.fact) + "</b>" + (r.calloutOnly ? ' <span class="badge neutral">خارج از جدول</span>' : "") +
            '<details class="row-more"><summary>شاهد آماری و اثر</summary><div class="body"><dl class="kv">' +
            "<dt>شاهد آماری</dt><dd>" + esc(r.evidence) + "</dd>" +
            (r.effect ? "<dt>اندازه‌ی اثر</dt><dd>" + esc(r.effect) + "</dd>" : "") +
            (r.impact ? "<dt>اثر بر پروژه</dt><dd>" + esc(r.impact) + "</dd>" : "") +
            "<dt>بخش</dt><dd>" + esc(r.section) + "</dd></dl></div></details>"; } },
      { key: "effect", title: "اندازه‌ی اثر", render: function (r) { return esc(r.effect || "—"); } },
      { key: "section", title: "بخش", render: function (r) {
          var p = splitSection(r.section);
          return '<span class="badge neutral" title="' + esc(r.section) + '">' + esc(p.letter || p.rest.slice(0, 10)) + "</span>"; } },
    ];
    return chips + sortableTable("facts", cols, rows, { sortKey: "id" });
  }

  /* ----------------------------------------------------------- پنل: یافته‌ها */

  function panelFindings() {
    var STATUS = [["", "همه"], ["confirmed", "تأییدشده"], ["rejected", "ردشده"], ["resolved", "رفع‌شده"], ["open", "باز"]];
    var phases = [];
    D.findings.forEach(function (f) { if (f.phase && phases.indexOf(f.phase) === -1) phases.push(f.phase); });
    var rows = D.findings.filter(function (f) {
      if (S.findStatus && f.statusKind !== S.findStatus) return false;
      if (S.findPhase && f.phase !== S.findPhase) return false;
      return match([f.id, f.finding, f.evidence, f.status, f.crossref].join(" "), S.findQ);
    });
    var badgeOf = { confirmed: "ok", rejected: "bad", resolved: "info", open: "warn", other: "neutral" };

    var chips = '<div class="chip-row"><input class="filter-input" id="findQ" placeholder="جست‌وجو در یافته‌ها…" value="' + esc(S.findQ) + '">' +
      STATUS.map(function (s) { return '<button class="chip' + (S.findStatus === s[0] ? " on" : "") + '" data-fstat="' + s[0] + '">' + s[1] + "</button>"; }).join("") +
      '<span style="color:var(--faint);font-size:12px">فاز:</span>' +
      phases.map(function (p) { return '<button class="chip' + (S.findPhase === p ? " on" : "") + '" data-fph="' + esc(p) + '">' + esc(p) + "</button>"; }).join("") +
      '<span class="result-count">' + fa(rows.length) + " از " + fa(D.findings.length) + "</span></div>";

    var cols = [
      { key: "sort", title: "شناسه", render: function (r) { return '<span class="badge id neutral">' + fa(r.id) + "</span>"; } },
      { key: "finding", title: "یافته", wide: true, render: function (r) {
          return esc(r.finding) + '<details class="row-more"><summary>شاهد و ارجاع</summary><div class="body"><dl class="kv">' +
            "<dt>شاهد</dt><dd>" + srcLink(r.evidence) + "</dd><dt>ارجاع متقابل</dt><dd>" + esc(r.crossref || "—") +
            "</dd><dt>گروه</dt><dd>" + esc(r.group) + "</dd></dl></div></details>"; } },
      { key: "phase", title: "فاز", render: function (r) { return '<span class="badge neutral">' + esc(r.phase) + "</span>"; } },
      { key: "status", title: "وضعیت", render: function (r) { return '<span class="badge ' + (badgeOf[r.statusKind] || "neutral") + '">' + esc(r.status) + "</span>"; } },
    ];
    return chips + sortableTable("findings", cols, rows, { sortKey: "sort" });
  }

  /* ---------------------------------------------------------- پنل: تصمیم‌ها */

  function panelDecisions() {
    var rows = D.decisions.filter(function (d) {
      return match([d.label, d.title, d.options, d.reason, d.effect, d.date].join(" "), S.decQ);
    });
    var chips = '<div class="chip-row"><input class="filter-input" id="decQ" placeholder="جست‌وجو در تصمیم‌ها…" value="' + esc(S.decQ) + '">' +
      '<span class="result-count">' + fa(rows.length) + " از " + fa(D.decisions.length) + "</span></div>";
    var cols = [
      { key: "id", title: "#", render: function (r) { return '<span class="badge id ' + (r.collision ? "warn" : "neutral") + '">' + fa(r.label) + "</span>"; } },
      { key: "title", title: "تصمیم", wide: true, render: function (r) {
          return "<b>" + esc(r.title) + "</b>" +
            '<details class="row-more"><summary>گزینه‌ها، دلیل و اثر</summary><div class="body"><dl class="kv">' +
            "<dt>گزینه‌های بررسی‌شده</dt><dd>" + esc(r.options) + "</dd>" +
            "<dt>دلیل انتخاب</dt><dd>" + esc(r.reason) + "</dd>" +
            "<dt>اثر بر نتیجه</dt><dd>" + esc(r.effect) + "</dd>" +
            (r.deepDive ? "<dt>تحلیل عمیق</dt><dd>" + srcLink(r.deepDive) + "</dd>" : "") +
            "</dl></div></details>"; } },
      { key: "date", title: "تاریخ", num: true, render: function (r) { return esc(r.date); } },
    ];
    return chips + sortableTable("decisions", cols, rows, { sortKey: "id" });
  }

  /* ------------------------------------------------------------ پنل: فیچرها */

  function panelFeatures() {
    var fams = [];
    D.features.forEach(function (f) { var k = f.family.replace(/[⭐⚠️]/g, "").trim(); if (fams.indexOf(k) === -1) fams.push(k); });
    var rows = D.features.filter(function (f) {
      if (S.featFamily && f.family.replace(/[⭐⚠️]/g, "").trim() !== S.featFamily) return false;
      return match([f.name, f.family, f.evidence, f.timing, f.sets].join(" "), S.featQ);
    });
    var chips = '<div class="chip-row"><input class="filter-input" id="featQ" placeholder="جست‌وجو در فیچرها…" value="' + esc(S.featQ) + '">' +
      '<button class="chip' + (S.featFamily ? "" : " on") + '" data-ffam="">همه</button>' +
      fams.map(function (f) { return '<button class="chip' + (S.featFamily === f ? " on" : "") + '" data-ffam="' + esc(f) + '">' + esc(f) + "</button>"; }).join("") +
      '<span class="result-count">' + fa(rows.length) + " از " + fa(D.features.length) + "</span></div>";
    var cols = [
      { key: "name", title: "فیچر", render: function (r) { return code(r.name) + (r.starred ? ' <span class="starred">⭐</span>' : "") + (r.risky ? ' <span title="وابسته به مقدار واقعی روز d">⚠️</span>' : ""); } },
      { key: "family", title: "خانواده", render: function (r) { return '<span class="badge neutral">' + esc(r.family.replace(/[⭐⚠️]/g, "").trim()) + "</span>"; } },
      { key: "timing", title: "وابستگی زمانی", wide: true, render: function (r) { return esc(r.timing); } },
      { key: "evidence", title: "شاهد", render: function (r) { return esc(r.evidence); } },
      { key: "coverage", title: "پوشش", num: true, sortVal: function (r) { return parseFloat(r.coverage) || 0; } },
      { key: "corr", title: "r با هدف", num: true, sortVal: function (r) { return parseFloat(r.corr) || 0; } },
      { key: "sets", title: "فیچرست‌ها", render: function (r) { return '<span style="font-size:12px;color:var(--muted)">' + esc(r.sets) + "</span>"; } },
    ];
    return chips + sortableTable("features", cols, rows, { sortKey: "name", compact: true }) +
      '<div class="note" style="margin-top:12px">قاعده‌ی حاکم فاز ۵: هیچ فیچری ساخته نشده که به یک ردیف دفتر حقایق یا دانش دامنه‌ای صریح متصل نباشد — ستون «شاهد» همان اتصال است. ⚠️ یعنی فیچر در استقرار باید <em>پیش‌بینی</em> شود، نه مقدار واقعی روز d.</div>';
  }

  /* ------------------------------------------------------------- پنل: مدل‌ها */

  function panelModels() {
    var all = D.models.comparison;
    var rows = all.filter(function (m) {
      if (S.modelWin && !m.beatsB3) return false;
      return match([m.model, m.family, m.level, m.compute].join(" "), S.modelQ);
    });
    var ch = all[0] || {};
    var B3 = 0.01335;

    var html = '<div class="grid k4">' +
      '<div class="kpi"><div class="label">قهرمان مطلق</div><div class="value sm">' + code(ch.model) + '</div><div class="foot">pinball نرخ ' + num(ch.pinball, 5) + "</div></div>" +
      '<div class="kpi"><div class="label">مرجع B3 (چندک تجربی)</div><div class="value sm">' + num(B3, 5) + '</div><div class="foot">Δ قهرمان: ' + num(ch.delta, 5) + "</div></div>" +
      '<div class="kpi"><div class="label">برد آماری بر B3</div><div class="value">' + fa(all.filter(function (m) { return m.beatsB3; }).length) + ' از ' + fa(all.length) + '</div><div class="foot">بوت‌استرپ بلوکی دوبعدی / DM-test</div></div>' +
      '<div class="kpi"><div class="label">اندازه‌ی مؤثر نمونه</div><div class="value">۳۳۲</div><div class="foot">از ۳٬۸۸۰ ردیف خام · ICC(روز)=۰٫۲۲۵ (F10)</div></div></div>';

    html += '<div class="card"><h3>Δ نسبت به خط پایه‌ی B3 (چندک تجربی)</h3>' +
      '<div class="card-note">مقدار منفی یعنی pinball کمتر از B3، یعنی بهتر. سبز پررنگ = برد آماری تأییدشده؛ ' +
      'آبی = بهتر ولی نه معنادار؛ قرمز = بدتر از B3. مرجع B3 = ' + num(B3, 5) + ' · τ=۰٫۲۰ · ' +
      '<span class="src">reports/phase7/model_comparison.csv</span></div>' +
      hbar(all.map(function (m) {
        return {
          label: code(m.model) + (m.compute !== "local" ? " ⚡" : ""),
          value: m.delta,
          text: (m.delta > 0 ? "+" : "") + num(m.delta, 5),
          color: m.beatsB3 ? C.ok : (m.delta < 0 ? C.accent : C.bad),
          dim: !m.beatsB3 && m.delta > 0,
        };
      }), { diverging: true, rawLabel: true }) +
      '<div class="legend"><span><i style="background:var(--ok)"></i>برد آماری تأییدشده</span>' +
      '<span><i style="background:var(--accent)"></i>بهتر از B3، بدون تأیید آماری</span>' +
      '<span><i style="background:var(--bad)"></i>بدتر از B3</span></div></div>';

    var chips = '<div class="chip-row"><input class="filter-input" id="modelQ" placeholder="جست‌وجو در مدل‌ها…" value="' + esc(S.modelQ) + '">' +
      '<button class="chip' + (S.modelWin ? " on" : "") + '" data-mwin="1">فقط بردهای معنادار</button>' +
      '<span class="result-count">' + fa(rows.length) + " از " + fa(all.length) + "</span></div>";

    var cols = [
      { key: "model", title: "مدل", render: function (r) { return code(r.model) + (r.compute !== "local" ? ' <span title="اجرای GPU">⚡</span>' : ""); } },
      { key: "family", title: "خانواده" },
      { key: "level", title: "سطح", render: function (r) { return '<span class="badge neutral">' + esc(r.level) + "</span>"; } },
      { key: "pinball", title: "pinball نرخ", num: true, render: function (r) { return num(r.pinball, 5); } },
      { key: "pinballPortions", title: "pinball پرس", num: true, render: function (r) { return num(r.pinballPortions, 3); } },
      { key: "delta", title: "Δ نسبت به B3", num: true, render: function (r) {
          var cls = r.delta < 0 ? "ok" : "bad";
          return '<span class="badge ' + cls + '">' + (r.delta > 0 ? "+" : "") + num(r.delta, 5) + "</span>"; } },
      { key: "beatsB3", title: "برد معنادار", render: function (r) { return r.beatsB3 ? '<span class="badge ok">✔</span>' : '<span class="badge neutral">—</span>'; }, sortVal: function (r) { return r.beatsB3 ? 1 : 0; } },
      { key: "wasteReduction", title: "هدررفت‌کاهی", num: true, render: function (r) { return pct(r.wasteReduction, 1); } },
      { key: "shortage", title: "نرخ کمبود", num: true, render: function (r) { return pct(r.shortage, 1); } },
      { key: "coverage", title: "پوشش", num: true, render: function (r) { return pct(r.coverage, 1); } },
      { key: "r2", title: "R²", num: true, render: function (r) { return num(r.r2, 3); } },
      { key: "fitSeconds", title: "ثانیه", num: true, render: function (r) { return num(r.fitSeconds, 1); } },
    ];
    html += chips + sortableTable("models", cols, rows, { sortKey: "pinball", compact: true });

    html += '<div class="note warn" style="margin-top:12px"><strong>هشدار قابلیت مقایسه:</strong> ستون‌های هدررفت‌کاهی، نرخ کمبود و R² برای ردیف‌های ⚡ (GPU) روی پایه‌ی متفاوتی حساب شده‌اند (میانگین fold، نه میانگین ردیفی) و مستقیماً با ردیف‌های CPU قابل‌قیاس نیستند. تنها ستون‌های بین‌گروهی معتبر: pinball نرخ، پوشش، و Δ نسبت به B3.</div>';

    html += '<div class="card"><h3>کارت مدل‌ها</h3><div class="card-note">' + fa(D.models.cards.length) + ' کارت در <span class="src">reports/models/</span> — هر کارت ۱۴ گام استاندارد فاز ۷ را دارد</div>' +
      '<div class="chip-row">' + D.models.cards.map(function (c) {
        return '<span class="chip" title="' + esc(c.file) + '">' + esc(c.name) + "</span>";
      }).join("") + "</div></div>";
    return html;
  }

  /* ---------------------------------------------------------- پنل: ارزیابی */

  function panelEvaluation() {
    var E = D.evaluation;
    var html = '<div class="note" style="margin-bottom:14px"><strong>پنجره‌ی Test فقط یک‌بار باز شد:</strong> ۱۴۰۳-۰۲-۰۳ تا ۱۴۰۳-۰۳-۰۱ (میلادی ۲۰۲۴-۰۴-۲۳ تا ۲۰۲۴-۰۵-۲۱) — رژیم غیرعادی رمضان و سوگواری. اندازه‌ی خام ۱٬۵۲۶ ردیف، اندازه‌ی مؤثر فقط ۱۰۵.</div>';

    html += '<div class="card"><h3>ارزیابی نهایی روی Test قفل‌شده</h3><div class="card-note">منبع: <span class="src">reports/phase8/8.1_holdout_metrics.csv</span></div>' +
      sortableTable("holdout", [
        { key: "model", title: "مدل", render: function (r) { return code(r.model); } },
        { key: "pinball", title: "pinball نرخ", num: true, render: function (r) { return num(r.pinball, 5); } },
        { key: "pinballPortions", title: "pinball پرس", num: true, render: function (r) { return num(r.pinballPortions, 3); } },
        { key: "rmse", title: "RMSE", num: true, render: function (r) { return num(r.rmse, 5); } },
        { key: "r2", title: "R²", num: true, render: function (r) { return num(r.r2, 3); } },
        { key: "coverage", title: "پوشش", num: true, render: function (r) { return pct(r.coverage, 1); } },
        { key: "coverageGap", title: "شکاف پوشش", num: true, render: function (r) { return '<span class="badge ' + (Math.abs(r.coverageGap) > .1 ? "warn" : "ok") + '">' + (r.coverageGap > 0 ? "+" : "") + pct(r.coverageGap, 1) + "</span>"; } },
        { key: "shortage", title: "نرخ کمبود", num: true, render: function (r) { return pct(r.shortage, 1); } },
        { key: "wasteReduction", title: "هدررفت‌کاهی", num: true, render: function (r) { return pct(r.wasteReduction, 1); } },
      ], E.holdout, { sortKey: "pinball", compact: true }) + "</div>";

    if (E.significance.length) {
      html += '<div class="card"><h3>مقایسه‌ی آماری در برابر B3</h3><div class="card-note">DM-test + بوت‌استرپ بلوکی دوبعدی روز×سلف · <span class="src">8.1_holdout_significance.csv</span></div>' +
        '<div class="table-wrap"><table class="table-compact"><thead><tr>' +
        Object.keys(E.significance[0]).map(function (k) { return '<th class="num">' + esc(k) + "</th>"; }).join("") +
        "</tr></thead><tbody>" + E.significance.map(function (r) {
          return "<tr>" + Object.keys(E.significance[0]).map(function (k) {
            var v = r[k];
            return '<td class="num">' + (typeof v === "number" ? num(v, 5) : esc(v)) + "</td>";
          }).join("") + "</tr>";
        }).join("") + "</tbody></table></div></div>";
    }

    // تجزیه‌ی خطا
    var bk = E.breakdown, keys = Object.keys(bk);
    if (keys.length) {
      if (keys.indexOf(S.breakdown) === -1) S.breakdown = keys[0];
      var cur = bk[S.breakdown];
      html += '<div class="card"><h3>تجزیه‌ی خطا</h3><div class="card-note">pinball و نرخ کمبود در هر برش · منبع: <span class="src">' + esc(cur.file) + "</span></div>" +
        '<div class="chip-row">' + keys.map(function (k) {
          return '<button class="chip' + (S.breakdown === k ? " on" : "") + '" data-bk="' + k + '">' + esc(bk[k].label) + "</button>";
        }).join("") + "</div>" +
        '<div class="grid k2"><div>' +
        "<div style=\"font-size:12.5px;color:var(--muted);margin-bottom:4px\">pinball (کمتر بهتر)</div>" +
        hbar(cur.rows.slice(0, 16).map(function (r) { return { label: r.name, value: r.pinball, text: num(r.pinball, 5) }; })) +
        "</div><div>" +
        "<div style=\"font-size:12.5px;color:var(--muted);margin-bottom:4px\">نرخ کمبود (کمتر بهتر)</div>" +
        hbar(cur.rows.slice(0, 16).map(function (r) { return { label: r.name, value: r.shortage, text: pct(r.shortage, 1), color: C.warn }; })) +
        "</div></div></div>";
    }

    // کالیبراسیون
    if (E.reliability.length) {
      html += '<div class="grid k2"><div class="card"><h3>نمودار اطمینان (کالیبراسیون)</h3>' +
        '<div class="card-note">پوشش تجربی در برابر پوشش اسمی · <span class="src">8.8_reliability_table.csv</span></div>' +
        lineChart([
          { name: "پوشش تجربی", color: "var(--accent)", points: E.reliability.map(function (r) { return [r.tau, r.coverage]; }) },
          { name: "پوشش اسمی (ایده‌آل)", color: "var(--faint)", points: E.reliability.map(function (r) { return [r.tau, r.tau]; }) },
        ], { width: 520, height: 230, xTicks: E.reliability.map(function (r) { return r.tau; }), xFmt: function (v) { return v.toFixed(2); }, yFmt: function (v) { return (v * 100).toFixed(0) + "%"; }, xLabel: "چندک اسمی τ" }) + "</div>";

      html += '<div class="card"><h3>حساسیت به seed روی مرز holdout</h3><div class="card-note"><span class="src">8.6_seed_sensitivity.csv</span></div>';
      if (E.seeds.length) {
        var sk = Object.keys(E.seeds[0]);
        html += '<div class="table-wrap"><table class="table-compact"><thead><tr>' + sk.map(function (k) { return '<th class="num">' + esc(k) + "</th>"; }).join("") + "</tr></thead><tbody>" +
          E.seeds.map(function (r) { return "<tr>" + sk.map(function (k) { return '<td class="num">' + (typeof r[k] === "number" ? num(r[k], 5) : esc(r[k])) + "</td>"; }).join("") + "</tr>"; }).join("") + "</tbody></table></div>";
      }
      html += "</div></div>";
    }
    return html;
  }

  /* ------------------------------------------------------------ پنل: تفسیر */

  function panelInterpretability() {
    var I = D.interpretability;
    var top = I.shap.slice(0, 18);
    var html = '<div class="grid k2"><div class="card"><h3>اهمیت SHAP — قهرمان روی L1</h3>' +
      '<div class="card-note">میانگین قدرمطلق SHAP · <span class="src">reports/phase9/9.2_shap_mean_abs.csv</span></div>' +
      hbar(top.map(function (r) { return { label: code(r.feature), value: r.shap, text: num(r.shap, 4) }; }), { rawLabel: true }) + "</div>";

    html += '<div class="card"><h3>اهمیت SHAP — سطح فرد (L5)</h3>' +
      '<div class="card-note"><span class="src">reports/phase9/9.6_l5_shap_mean_abs.csv</span></div>' +
      hbar(I.shapL5.slice(0, 14).map(function (r) { return { label: code(r.feature), value: r.shap, text: num(r.shap, 3), color: C.info }; }), { rawLabel: true }) + "</div></div>";

    if (I.groups.length) {
      html += '<div class="card"><h3>سهم گروه‌های فیچر در سطح فرد</h3>' +
        '<div class="card-note">توضیح مستقیم اینکه چرا L5 در بند ۸.۱۰ به L1/L2 نباخت (یافته‌ی ۴۶)</div>' +
        '<div class="table-wrap"><table><thead><tr><th>گروه</th><th class="num">تعداد فیچر</th><th class="num">مجموع |SHAP|</th><th class="num">سهم</th><th>فیچر برتر</th></tr></thead><tbody>' +
        I.groups.map(function (g) {
          return "<tr><td>" + esc(g.group) + '</td><td class="num">' + fa(g.nFeatures) + '</td><td class="num">' + num(g.sumShap, 4) +
            '</td><td class="num"><b>' + pctRaw(g.share, 1) + "</b></td><td>" + code(g.topFeature) + "</td></tr>";
        }).join("") + "</tbody></table></div></div>";
    }

    html += '<div class="card"><h3>اهمیت فیچر قهرمان (gain در برابر permutation)</h3>' +
      '<div class="card-note"><span class="src">reports/phase9/9.1_feature_importance.csv</span> — اختلاف رتبه‌ی این دو ستون خودش یک یافته است</div>' +
      sortableTable("imp", [
        { key: "feature", title: "فیچر", render: function (r) { return code(r.feature); } },
        { key: "gain", title: "gain", num: true, render: function (r) { return num(r.gain, 1); } },
        { key: "gainRank", title: "رتبه‌ی gain", num: true, render: function (r) { return fa(r.gainRank); } },
        { key: "permMean", title: "permutation", num: true, render: function (r) { return num(r.permMean, 6); } },
        { key: "permRank", title: "رتبه‌ی perm", num: true, render: function (r) { return fa(r.permRank); } },
        { key: "split", title: "split", num: true, render: function (r) { return fa(r.split); } },
      ], I.importance, { sortKey: "gainRank", compact: true }) + "</div>";
    return html;
  }

  /* ------------------------------------------------- پنل: لایه‌ی تصمیم (τ) */

  function panelDecisionLayer() {
    var L = D.decisionLayer, curve = L.tauCurve;
    if (!curve.length) return '<div class="empty-state">داده‌ی فاز ۱۰ پیدا نشد.</div>';

    // نزدیک‌ترین نقطه‌ی منحنی به τ انتخابی
    var pt = curve.reduce(function (best, r) {
      return Math.abs(r.tau - S.tau) < Math.abs(best.tau - S.tau) ? r : best;
    }, curve[0]);
    var net = L.netValue.filter(function (r) { return r.model !== "B3"; });
    var netPt = net.reduce(function (best, r) { return Math.abs(r.tau - pt.tau) < Math.abs(best.tau - pt.tau) ? r : best; }, net[0] || {});
    var scen = L.scenarios.reduce(function (best, r) { return Math.abs(r.tau - pt.tau) < Math.abs(best.tau - pt.tau) ? r : best; }, L.scenarios[0] || {});
    var exact = scen && Math.abs(scen.tau - pt.tau) < 1e-6;

    var html = '<div class="card"><h3>معاوضه‌ی چندک: اهرم سیاستی پروژه</h3>' +
      '<div class="card-note">τ ورودی سیاستی است، نه خروجی مدل. τ* = C₀/(C_u+C₀) — ردیف ۳۶ <span class="src">decision_log</span>. نقطه‌ی عملیاتی پروژه τ=۰٫۲۰.</div>' +
      '<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:14px">' +
      '<label style="font-size:13.5px;color:var(--muted)">چندک τ</label>' +
      '<input type="range" id="tauSlider" min="0" max="' + (curve.length - 1) + '" step="1" value="' + curve.indexOf(pt) + '" style="flex:1;min-width:240px">' +
      '<b style="font-family:var(--mono);font-size:19px;direction:ltr">τ = ' + num(pt.tau, 2) + "</b></div>";

    html += '<div class="grid k4">' +
      '<div class="kpi"><div class="label">کاهش هدررفت</div><div class="value">' + pctRaw(pt.wasteReduction, 1) + '</div><div class="foot">نسبت به پخت بدون مدل</div></div>' +
      '<div class="kpi"><div class="label">نرخ کمبود سلول</div><div class="value">' + pct(pt.shortage, 1) + '</div><div class="foot">فقط <em>رخداد</em> را می‌شمارد، نه عمق</div></div>' +
      '<div class="kpi"><div class="label">پرس صرفه‌جویی‌شده</div><div class="value">' + group(pt.portionsSaved) + '</div><div class="foot">روی ۸۰ روز سرو foldهای رسمی</div></div>' +
      '<div class="kpi"><div class="label">ارزش خالص</div><div class="value">' + milyard(netPt.netValue) + ' میلیارد</div><div class="foot">هدررفت‌کاهی منهای هزینه‌ی کمبود</div></div></div>';

    if (exact) {
      html += '<div class="note" style="margin-top:14px"><strong>سناریوی «' + esc(scen.scenario) + '»:</strong> ' +
        'کمبود ' + fa(num(scen.shortagePerMeal, 1)) + ' پرس در هر وعده (میانه ' + fa(num(scen.medianPerMeal, 1)) + '، بدترین وعده‌ی کل دوره ' + fa(num(scen.worstMeal, 0)) + ') · ' +
        'کسر تقاضای برآورده‌نشده ' + pct(scen.unmetRate, 2) + ' یعنی Fill Rate ' + pct(scen.fillRate, 2) + ' · ' +
        'صرفه‌جویی سالانه <b>' + milyard(scen.annualToman) + ' میلیارد تومان</b> [CI ۹۵٪: ' + milyard(scen.annualTomanLo) + " تا " + milyard(scen.annualTomanHi) + "]</div>";
    }
    html += "</div>";

    var ticks = curve.map(function (r) { return r.tau; });
    html += '<div class="grid k2"><div class="card"><h3>هدررفت‌کاهی در برابر نرخ کمبود</h3>' +
      '<div class="card-note"><span class="src">reports/phase10/10.7_tau_curve.csv</span></div>' +
      lineChart([
        { name: "کاهش هدررفت", color: "var(--ok)", points: curve.map(function (r) { return [r.tau, r.wasteReduction / 100]; }) },
        { name: "نرخ کمبود سلول", color: "var(--bad)", points: curve.map(function (r) { return [r.tau, r.shortage]; }) },
      ], { width: 520, height: 250, xTicks: ticks, xFmt: function (v) { return v.toFixed(2); }, yFmt: function (v) { return (v * 100).toFixed(0) + "%"; }, xLabel: "چندک τ", markX: pt.tau }) + "</div>";

    html += '<div class="card"><h3>ارزش خالص تومانی در برابر τ</h3>' +
      '<div class="card-note">قهرمان در برابر خط پایه‌ی B3 · <span class="src">10.2_net_value_by_tau.csv</span></div>' +
      lineChart([
        { name: "قهرمان", color: "var(--accent)", points: net.map(function (r) { return [r.tau, r.netValue / 1e9]; }) },
        { name: "B3", color: "var(--faint)", points: L.netValue.filter(function (r) { return r.model === "B3"; }).map(function (r) { return [r.tau, r.netValue / 1e9]; }) },
      ], { width: 520, height: 250, xTicks: net.map(function (r) { return r.tau; }), xFmt: function (v) { return v.toFixed(2); }, yFmt: function (v) { return v.toFixed(1); }, xLabel: "چندک τ  (محور عمودی: میلیارد تومان)", markX: pt.tau }) + "</div></div>";

    html += '<div class="card"><h3>سه سناریوی سیاستی</h3><div class="card-note"><span class="src">reports/phase10/10.3_policy_scenarios.csv</span></div>' +
      '<div class="table-wrap"><table><thead><tr><th>سناریو</th><th class="num">τ</th><th class="num">نرخ کمبود</th><th class="num">کمبود در هر وعده</th><th class="num">بدترین وعده</th><th class="num">Fill Rate</th><th class="num">صرفه‌جویی سالانه (میلیارد تومان)</th></tr></thead><tbody>' +
      L.scenarios.map(function (s) {
        return "<tr><td><b>" + esc(s.scenario) + '</b></td><td class="num">' + num(s.tau, 2) + '</td><td class="num">' + pct(s.shortage, 1) +
          '</td><td class="num">' + fa(num(s.shortagePerMeal, 1)) + '</td><td class="num">' + fa(num(s.worstMeal, 0)) +
          '</td><td class="num">' + pct(s.fillRate, 2) + '</td><td class="num"><b>' + milyard(s.annualToman) + "</b> [" + milyard(s.annualTomanLo) + "، " + milyard(s.annualTomanHi) + "]</td></tr>";
      }).join("") + "</tbody></table></div>" +
      '<div class="note warn" style="margin-top:12px"><strong>قاعده‌ی صداقت:</strong> هر عدد ترسناک با مقیاس‌گذارش هم‌زمان دیده شود. «نرخ کمبود ۱۵٫۴٪» فقط <em>رخداد</em> کمبود در سلول را می‌شمارد؛ کسر تقاضای برآورده‌نشده در همان نقطه ۰٫۲۸٪ است (Fill Rate ۹۹٫۷۲٪). مبنای همه‌ی اعداد تومانی: ۳۲ هفته‌ی آموزشی، ۳۷۵٬۰۰۰ تومان هر پرس (شهریور ۱۴۰۵) — به تابستان، تعطیلات و رشد ثبت‌نام تعمیم‌پذیر نیست.</div></div>';
    return html;
  }

  /* ------------------------------------------------------------- پنل: شکل‌ها */

  function panelFigures() {
    var groups = [];
    D.figures.forEach(function (f) { if (groups.indexOf(f.group) === -1) groups.push(f.group); });
    var rows = D.figures.filter(function (f) {
      if (S.figGroup && f.group !== S.figGroup) return false;
      return match(f.name + " " + f.group, S.figQ);
    });
    var chips = '<div class="chip-row"><input class="filter-input" id="figQ" placeholder="جست‌وجو در شکل‌ها…" value="' + esc(S.figQ) + '">' +
      '<button class="chip' + (S.figGroup ? "" : " on") + '" data-fig="">همه</button>' +
      groups.map(function (g) { return '<button class="chip' + (S.figGroup === g ? " on" : "") + '" data-fig="' + esc(g) + '">' + esc(g) + "</button>"; }).join("") +
      '<span class="result-count">' + fa(rows.length) + " از " + fa(D.figures.length) + "</span></div>";
    var grid = '<div class="fig-grid">' + rows.map(function (f) {
      return '<figure class="fig" data-src="' + esc(f.path) + '" data-name="' + esc(f.repoPath) + '" style="margin:0">' +
        '<img loading="lazy" src="' + esc(f.path) + '" alt="' + esc(f.name) + '">' +
        '<figcaption class="cap"><b>' + esc(f.name) + "</b><small>" + esc(f.group) + " · " + fa(f.sizeKB) + " کیلوبایت" + (f.persian ? " · نسخه‌ی فارسی" : "") + "</small></figcaption></figure>";
    }).join("") + "</div>";
    if (!rows.length) grid = '<div class="empty-state">شکلی با این فیلتر پیدا نشد.</div>';
    return chips + grid;
  }

  /* -------------------------------------------------------------- پنل: اسناد */

  function panelDocs() {
    var groups = [];
    D.docs.forEach(function (d) { if (groups.indexOf(d.group) === -1) groups.push(d.group); });
    var rows = D.docs.filter(function (d) {
      if (S.docGroup && d.group !== S.docGroup) return false;
      return match(d.path + " " + d.title + " " + d.note, S.docQ);
    });
    var chips = '<div class="chip-row"><input class="filter-input" id="docQ" placeholder="جست‌وجو در اسناد…" value="' + esc(S.docQ) + '">' +
      '<button class="chip' + (S.docGroup ? "" : " on") + '" data-doc="">همه</button>' +
      groups.map(function (g) { return '<button class="chip' + (S.docGroup === g ? " on" : "") + '" data-doc="' + esc(g) + '">' + esc(g) + "</button>"; }).join("") +
      '<span class="result-count">' + fa(rows.length) + " از " + fa(D.docs.length) + "</span></div>";
    var cols = [
      { key: "path", title: "مسیر", render: function (r) { return '<a href="../' + esc(r.path) + '" class="src" style="color:var(--accent)">' + esc(r.path) + "</a>"; } },
      { key: "title", title: "عنوان", wide: true },
      { key: "note", title: "نقش", render: function (r) { return '<span style="color:var(--muted)">' + esc(r.note || "—") + "</span>"; } },
      { key: "lines", title: "خط", num: true, render: function (r) { return fa(r.lines); } },
      { key: "sizeKB", title: "کیلوبایت", num: true, render: function (r) { return fa(r.sizeKB); } },
    ];
    return chips + sortableTable("docs", cols, rows, { sortKey: "path", compact: true });
  }

  /* ---------------------------------------------------------------- تب‌ها */

  var TABS = [
    { id: "overview", title: "نمای کلی", render: panelOverview, lede: "لنگرهای عددی پروژه، پیشرفت فازها و دروازه‌ها در یک صفحه." },
    { id: "phases", title: "فازها", render: panelPhases, count: function () { return D.meta.counts.phaseItems; }, lede: "وضعیت هر بسته‌ی کاری، مستقیماً از فایل‌های doc/progress/. روی هر کارت بزنید تا آیتم‌ها باز شوند." },
    { id: "data", title: "داده و قواعد", render: panelData, lede: "سطوح داده، قاعده‌ی برش اطلاعاتی و اسناد پایه." },
    { id: "facts", title: "حقایق EDA", render: panelFacts, count: function () { return D.meta.counts.facts; }, lede: "دفتر حقایق فاز ۴ — خروجی دروازه‌ی M2. هیچ فیچری ساخته نشده که به یکی از این ردیف‌ها وصل نباشد." },
    { id: "findings", title: "یافته‌ها", render: panelFindings, count: function () { return D.meta.counts.findings; }, lede: "کشف‌های تفسیری و تصمیم‌ساز فازهای ۰، ۷، ۸، ۹ و ۱۰ — با وضعیت تأیید یا رد." },
    { id: "decisions", title: "تصمیم‌ها", render: panelDecisions, count: function () { return D.meta.counts.decisions; }, lede: "هر تصمیم روش‌شناختی با گزینه‌های بررسی‌شده و دلیل انتخاب. این جدول مستقیماً فصل روش‌شناسی گزارش است." },
    { id: "features", title: "فیچرها", render: panelFeatures, count: function () { return D.meta.counts.features; }, lede: "رجیستری خودکار فاز ۵ — با ستون شاهد که هر فیچر را به یک حقیقت وصل می‌کند." },
    { id: "models", title: "مدل‌ها", render: panelModels, count: function () { return D.meta.counts.models; }, lede: "جدول مقایسه‌ی نهایی فاز ۷ (دروازه‌ی M4) روی foldهای رسمی، τ=۰٫۲۰." },
    { id: "evaluation", title: "ارزیابی", render: panelEvaluation, lede: "فاز ۸ — عملکرد روی Test قفل‌شده، تجزیه‌ی خطا و کالیبراسیون." },
    { id: "interpretability", title: "تفسیرپذیری", render: panelInterpretability, lede: "فاز ۹ — مدل چه چیزی یاد گرفت، در سطح سلول و در سطح فرد." },
    { id: "decision-layer", title: "لایه‌ی تصمیم", render: panelDecisionLayer, lede: "فاز ۱۰ — از چندک تا مقدار پخت و تومان. τ را عوض کنید و پیامدش را ببینید." },
    { id: "figures", title: "شکل‌ها", render: panelFigures, count: function () { return D.meta.counts.figures; }, lede: "همه‌ی شکل‌های تولیدشده‌ی مخزن. روی هر شکل بزنید تا بزرگ شود." },
    { id: "docs", title: "اسناد", render: panelDocs, count: function () { return D.meta.counts.docs; }, lede: "نقشه‌ی کامل اسناد doc/ و reports/ با نقش هرکدام." },
  ];

  function render() {
    var nav = $("#sidenav"), main = $("#main");
    nav.innerHTML = '<div class="nav-group">بخش‌ها</div>' + TABS.map(function (t) {
      return '<a href="#' + t.id + '" class="' + (S.tab === t.id ? "active" : "") + '">' + esc(t.title) +
        (t.count ? '<span class="count">' + fa(t.count()) + "</span>" : "") + "</a>";
    }).join("");
    var tab = TABS.filter(function (t) { return t.id === S.tab; })[0] || TABS[0];
    main.innerHTML = '<div class="panel active"><div class="panel-head"><h2>' + esc(tab.title) + "</h2>" +
      (tab.lede ? '<div class="lede">' + esc(tab.lede) + "</div>" : "") + "</div>" + tab.render() + "</div>";
    document.title = tab.title + " — داشبورد دانش پروژه";
  }

  /* -------------------------------------------------------- جست‌وجوی سراسری */

  var INDEX = null;
  function buildIndex() {
    if (INDEX) return INDEX;
    INDEX = [];
    D.facts.forEach(function (f) { INDEX.push({ kind: "حقیقت " + f.id, tab: "facts", title: f.fact, sub: f.effect || f.section, text: [f.id, f.fact, f.evidence, f.effect, f.impact].join(" ") }); });
    D.findings.forEach(function (f) { INDEX.push({ kind: "یافته " + f.id, tab: "findings", title: f.finding, sub: f.status + " · فاز " + f.phase, text: [f.id, f.finding, f.evidence, f.crossref].join(" ") }); });
    D.decisions.forEach(function (d) { INDEX.push({ kind: "تصمیم " + d.label, tab: "decisions", title: d.title, sub: d.date, text: [d.label, d.title, d.options, d.reason, d.effect].join(" ") }); });
    D.features.forEach(function (f) { INDEX.push({ kind: "فیچر", tab: "features", title: f.name, sub: f.family + " · شاهد " + f.evidence, text: [f.name, f.family, f.evidence, f.sets].join(" ") }); });
    D.models.comparison.forEach(function (m) { INDEX.push({ kind: "مدل", tab: "models", title: m.model, sub: m.family + " · pinball " + num(m.pinball, 5), text: [m.model, m.family, m.level].join(" ") }); });
    D.figures.forEach(function (f) { INDEX.push({ kind: "شکل", tab: "figures", title: f.name, sub: f.group, text: f.name + " " + f.group }); });
    D.docs.forEach(function (d) { INDEX.push({ kind: "سند", tab: "docs", title: d.title, sub: d.path, text: [d.path, d.title, d.note].join(" ") }); });
    D.phases.forEach(function (p) {
      p.items.forEach(function (it) { INDEX.push({ kind: "بسته‌ی کاری فاز " + p.code, tab: "phases", title: it.text, sub: (it.done ? "انجام‌شده" : "باقی‌مانده") + " · " + it.group, text: it.text + " " + it.group }); });
    });
    return INDEX;
  }

  var searchSel = 0, searchRows = [];
  function runSearch(q) {
    var box = $("#searchResults");
    if (!q || q.length < 2) { box.innerHTML = '<div class="empty">حداقل دو نویسه بنویسید. جست‌وجو روی حقایق، یافته‌ها، تصمیم‌ها، فیچرها، مدل‌ها، شکل‌ها، اسناد و بسته‌های کاری اجرا می‌شود.</div>'; searchRows = []; return; }
    searchRows = buildIndex().filter(function (r) { return match(r.text, q); }).slice(0, 60);
    searchSel = 0;
    if (!searchRows.length) { box.innerHTML = '<div class="empty">نتیجه‌ای پیدا نشد.</div>'; return; }
    box.innerHTML = searchRows.map(function (r, i) {
      return '<button class="res' + (i === 0 ? " sel" : "") + '" data-i="' + i + '"><span class="kind">' + esc(r.kind) +
        '</span><span class="ttl">' + esc(String(r.title).slice(0, 130)) + '</span><span class="sub">' + esc(String(r.sub || "").slice(0, 110)) + "</span></button>";
    }).join("");
  }
  function openSearch() { $("#searchpanel").classList.add("on"); var i = $("#searchInput"); i.value = ""; runSearch(""); i.focus(); }
  function closeSearch() { $("#searchpanel").classList.remove("on"); }
  function gotoResult(i) {
    var r = searchRows[i]; if (!r) return;
    closeSearch();
    location.hash = "#" + r.tab;
  }

  /* ------------------------------------------------------------- رویدادها */

  function route() {
    var id = (location.hash || "#overview").slice(1);
    if (TABS.some(function (t) { return t.id === id; })) S.tab = id;
    render();
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", route);

  document.addEventListener("input", function (e) {
    var t = e.target;
    if (t.id === "searchInput") { runSearch(t.value); return; }
    var mapping = { factQ: "factQ", findQ: "findQ", decQ: "decQ", featQ: "featQ", modelQ: "modelQ", figQ: "figQ", docQ: "docQ" };
    if (mapping[t.id]) {
      S[mapping[t.id]] = t.value;
      var pos = t.selectionStart, id = t.id;
      render();
      var again = document.getElementById(id);
      if (again) { again.focus(); try { again.setSelectionRange(pos, pos); } catch (_) {} }
      return;
    }
    if (t.id === "tauSlider") {
      var curve = D.decisionLayer.tauCurve;
      S.tau = curve[+t.value].tau;
      render();
      var s = document.getElementById("tauSlider");
      if (s) s.focus();
    }
  });

  document.addEventListener("click", function (e) {
    var t = e.target;
    var fig = t.closest && t.closest(".fig");
    if (fig) {
      $("#lightbox").classList.add("on");
      $("#lightbox img").src = fig.dataset.src;
      $("#lightbox .lb-cap").textContent = fig.dataset.name;
      return;
    }
    if (t.closest && t.closest("#lightbox")) { $("#lightbox").classList.remove("on"); return; }
    var res = t.closest && t.closest(".res");
    if (res) { gotoResult(+res.dataset.i); return; }
    if (t.id === "searchBtn" || (t.closest && t.closest("#searchBtn"))) { openSearch(); return; }
    if (t.closest && t.closest("#searchpanel") && !t.closest(".sheet")) { closeSearch(); return; }
    if (t.id === "themeBtn") { toggleTheme(); return; }

    var chip = t.closest && t.closest(".chip");
    if (chip) {
      if (chip.dataset.fsec !== undefined) { S.factSection = chip.dataset.fsec; render(); return; }
      if (chip.dataset.fstar !== undefined) { S.factStar = !S.factStar; render(); return; }
      if (chip.dataset.fstat !== undefined) { S.findStatus = S.findStatus === chip.dataset.fstat ? "" : chip.dataset.fstat; render(); return; }
      if (chip.dataset.fph !== undefined) { S.findPhase = S.findPhase === chip.dataset.fph ? "" : chip.dataset.fph; render(); return; }
      if (chip.dataset.ffam !== undefined) { S.featFamily = chip.dataset.ffam; render(); return; }
      if (chip.dataset.mwin !== undefined) { S.modelWin = !S.modelWin; render(); return; }
      if (chip.dataset.fig !== undefined) { S.figGroup = chip.dataset.fig; render(); return; }
      if (chip.dataset.doc !== undefined) { S.docGroup = chip.dataset.doc; render(); return; }
      if (chip.dataset.bk !== undefined) { S.breakdown = chip.dataset.bk; render(); return; }
    }
    var pc = t.closest && t.closest(".phase-card");
    if (pc && !t.closest("a") && !t.closest("details")) { S.phaseOpen[pc.dataset.phase] = !S.phaseOpen[pc.dataset.phase]; render(); }
  });

  document.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openSearch(); return; }
    if (e.key === "/" && document.activeElement.tagName !== "INPUT") { e.preventDefault(); openSearch(); return; }
    if (e.key === "Escape") { closeSearch(); $("#lightbox").classList.remove("on"); return; }
    if (!$("#searchpanel").classList.contains("on")) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      var items = $$("#searchResults .res");
      if (!items.length) return;
      items[searchSel] && items[searchSel].classList.remove("sel");
      searchSel = (searchSel + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items[searchSel].classList.add("sel");
      items[searchSel].scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter") { e.preventDefault(); gotoResult(searchSel); }
  });

  function toggleTheme() {
    var cur = document.documentElement.getAttribute("data-theme");
    var next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("dash-theme", next); } catch (_) {}
    $("#themeBtn").textContent = next === "dark" ? "☀ روشن" : "🌙 تیره";
  }

  /* ------------------------------------------------------------------ شروع */

  (function init() {
    var saved = null;
    try { saved = localStorage.getItem("dash-theme"); } catch (_) {}
    var theme = saved || (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", theme);
    $("#themeBtn").textContent = theme === "dark" ? "☀ روشن" : "🌙 تیره";
    $("#buildMeta").innerHTML = 'ساخت <bdi class="code">' + esc(D.meta.builtAt) + '</bdi> · <bdi class="code">' + esc(D.meta.gitCommit) + "</bdi>";
    route();
  })();
})();
