"use client";

import { useState } from "react";

const tabs = [
  { id: "overview", label: "Overview" },
  { id: "problem", label: "Problem" },
  { id: "solution", label: "Solution" },
  { id: "results", label: "Results" },
  { id: "future", label: "Future" },
];

// Video tiles for overview page - 50 tiles for 10-col grid (5 rows)
const videoTiles = [
  { src: "/videos/signs/hello.mp4", label: "HELLO" },
  { src: "/videos/signs/thank_you.mp4", label: "THANK YOU" },
  { src: "/videos/signs/please.mp4", label: "PLEASE" },
  { src: "/videos/signs/help.mp4", label: "HELP" },
  { src: "/videos/signs/yes.mp4", label: "YES" },
  { src: "/videos/signs/no.mp4", label: "NO" },
  { src: "/videos/signs/good.mp4", label: "GOOD" },
  { src: "/videos/signs/bad.mp4", label: "BAD" },
  { src: "/videos/signs/family.mp4", label: "FAMILY" },
  { src: "/videos/signs/friend.mp4", label: "FRIEND" },
  { src: "/videos/signs/love.mp4", label: "LOVE" },
  { src: "/videos/signs/happy.mp4", label: "HAPPY" },
  { src: "/videos/signs/sad.mp4", label: "SAD" },
  { src: "/videos/signs/work.mp4", label: "WORK" },
  { src: "/videos/signs/school.mp4", label: "SCHOOL" },
  { src: "/videos/signs/eat.mp4", label: "EAT" },
  { src: "/videos/signs/drink.mp4", label: "DRINK" },
  { src: "/videos/signs/water.mp4", label: "WATER" },
  { src: "/videos/signs/home.mp4", label: "HOME" },
  { src: "/videos/signs/mother.mp4", label: "MOTHER" },
  { src: "/videos/signs/father.mp4", label: "FATHER" },
  { src: "/videos/signs/sorry.mp4", label: "SORRY" },
  { src: "/videos/signs/want.mp4", label: "WANT" },
  { src: "/videos/signs/need.mp4", label: "NEED" },
  { src: "/videos/signs/like.mp4", label: "LIKE" },
  { src: "/videos/signs/know.mp4", label: "KNOW" },
  { src: "/videos/signs/see.mp4", label: "SEE" },
  { src: "/videos/signs/think.mp4", label: "THINK" },
  { src: "/videos/signs/big.mp4", label: "BIG" },
  { src: "/videos/signs/small.mp4", label: "SMALL" },
  { src: "/videos/signs/where.mp4", label: "WHERE" },
  { src: "/videos/signs/who.mp4", label: "WHO" },
  { src: "/videos/signs/when.mp4", label: "WHEN" },
  { src: "/videos/signs/how.mp4", label: "HOW" },
  { src: "/videos/signs/name.mp4", label: "NAME" },
  { src: "/videos/signs/sleep.mp4", label: "SLEEP" },
  { src: "/videos/signs/play.mp4", label: "PLAY" },
  { src: "/videos/signs/learn.mp4", label: "LEARN" },
  { src: "/videos/signs/feel.mp4", label: "FEEL" },
  { src: "/videos/signs/funny.mp4", label: "FUNNY" },
  { src: "/videos/signs/give.mp4", label: "GIVE" },
  { src: "/videos/signs/make.mp4", label: "MAKE" },
  { src: "/videos/signs/open.mp4", label: "OPEN" },
  { src: "/videos/signs/close.mp4", label: "CLOSE" },
  { src: "/videos/signs/start.mp4", label: "START" },
  { src: "/videos/signs/stop.mp4", label: "STOP" },
  { src: "/videos/signs/wait.mp4", label: "WAIT" },
  { src: "/videos/signs/finish.mp4", label: "FINISH" },
  { src: "/videos/signs/try.mp4", label: "TRY" },
  { src: "/videos/signs/find.mp4", label: "FIND" },
];

// ============================================================================
// PAGE 1: OVERVIEW
// ============================================================================
function OverviewPage() {
  return (
    <div className="relative">
      {/* Video Background - denser grid, more visible */}
      <div className="absolute inset-0 z-0 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-white/80 via-white/65 to-white/80 z-10" />
        <div className="grid grid-cols-10 gap-0.5 opacity-75">
          {videoTiles.map((v, i) => (
            <div key={i} className="aspect-[3/4] relative overflow-hidden rounded-sm bg-gray-200" style={{ contain: "layout style paint" }}>
              <video src={v.src} autoPlay loop muted playsInline className="w-full h-full object-cover" />
              <div className="absolute bottom-0 inset-x-0 bg-black/70 text-white text-[5px] font-bold text-center py-px">
                {v.label}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="relative z-20 p-5 space-y-3">
        {/* Header */}
        <div className="text-center">
          <div className="inline-block bg-[#2D5A4A] text-white text-xs font-bold px-4 py-1 rounded-full mb-1">
            Toshiba Challenge 2026
          </div>
          <h1 className="text-4xl font-serif text-gray-900">SignSense</h1>
          <p className="text-gray-600 mt-1 max-w-xl mx-auto text-sm">
            AI-powered American Sign Language learning with <strong>real-time diagnostic feedback</strong> —
            not just right or wrong, but exactly what to fix.
          </p>
        </div>

        {/* Four Models + Benefits */}
        <div className="grid grid-cols-4 gap-3">
          {/* Benefits - slightly taller with more padding */}
          <div className="space-y-2">
            {[
              { icon: "🏆", title: "Highest Accuracy", desc: "88.4% on WLASL100 — best skeleton-only result ever published" },
              { icon: "🔒", title: "Privacy First", desc: "Only skeleton data processed — your video never leaves your device" },
              { icon: "💬", title: "Real Feedback", desc: "Tells you exactly what to fix, not just right or wrong" },
            ].map((b) => (
              <div key={b.title} className="bg-white/95 backdrop-blur rounded-lg p-3 shadow-sm border-l-4 border-[#2D5A4A]">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-base">{b.icon}</span>
                  <span className="font-bold text-xs text-gray-900">{b.title}</span>
                </div>
                <p className="text-[11px] text-gray-600 leading-relaxed">{b.desc}</p>
              </div>
            ))}
          </div>

          {/* Four Models - spans 3 columns, slightly taller */}
          <div className="col-span-3 bg-white/90 backdrop-blur rounded-xl p-5 shadow-lg flex flex-col">
            <h2 className="text-center text-sm font-bold text-gray-800 mb-3">Four-Model Diagnostic Pipeline</h2>
            <div className="grid grid-cols-4 gap-3 flex-1">
              {[
                { n: "1", name: "PhonSSM", desc: "Sign recognition using phonological decomposition", stat: "88.4% accuracy", color: "#2D5A4A" },
                { n: "2", name: "Error Diagnosis", desc: "Identifies specific component errors", stat: "16 error types", color: "#C75D4D" },
                { n: "3", name: "Movement", desc: "Evaluates motion quality and fluency", stat: "6 movement classes", color: "#E8B86D" },
                { n: "4", name: "Feedback Ranker", desc: "Prioritizes corrections for learning", stat: "TFLite optimized", color: "#7C5CBF" },
              ].map((m) => (
                <div key={m.n} className="text-center">
                  <div className="w-10 h-10 rounded-full flex items-center justify-center mx-auto mb-1.5 text-white text-lg font-bold" style={{ backgroundColor: m.color }}>
                    {m.n}
                  </div>
                  <h3 className="font-bold text-xs text-gray-900">{m.name}</h3>
                  <p className="text-[10px] text-gray-600 mt-0.5">{m.desc}</p>
                  <div className="mt-1.5 inline-block bg-gray-100 text-[10px] font-medium px-2 py-0.5 rounded" style={{ color: m.color }}>
                    {m.stat}
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 pt-3 border-t border-gray-200">
              <p className="text-xs text-gray-700 text-center leading-relaxed">
                A revolutionary <strong>privacy-first</strong>, real-time, AI-powered sign language education platform for everyone.
                Built on the largest ASL dataset ever assembled (5,565 signs across 4 datasets), powered by <strong>PhonSSM</strong> — a novel
                deep learning architecture that decomposes signs into phonological components for targeted feedback.
                Achieves the <strong>highest accuracy ever recorded</strong> on all WLASL benchmarks (skeleton-only).
              </p>
            </div>
          </div>
        </div>

        {/* Benchmark + Pipeline + Stats in one row */}
        <div className="grid grid-cols-5 gap-3">
          {/* Benchmark Results */}
          <div className="col-span-2 bg-[#2D5A4A] text-white rounded-xl p-3">
            <h3 className="text-xs font-bold mb-2 opacity-80">WLASL BENCHMARK (vs previous skeleton-only best)</h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left opacity-70 text-xs">
                  <th className="pb-1">Dataset</th>
                  <th className="pb-1 text-right">Previous</th>
                  <th className="pb-1 text-right">Ours</th>
                  <th className="pb-1 text-right">Gain</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { d: "WLASL100", p: "63.2%", o: "88.4%", g: "+25.2%" },
                  { d: "WLASL300", p: "58.4%", o: "74.4%", g: "+16.0%" },
                  { d: "WLASL2000", p: "53.7%", o: "72.1%", g: "+18.4%" },
                ].map((r) => (
                  <tr key={r.d} className="border-t border-white/20">
                    <td className="py-1">{r.d}</td>
                    <td className="py-1 text-right opacity-70">{r.p}</td>
                    <td className="py-1 text-right font-bold">{r.o}</td>
                    <td className="py-1 text-right text-[#E8B86D] font-medium">{r.g}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Application Images */}
          <div className="col-span-2 grid grid-cols-2 gap-2">
            {[
              { src: "/images/applications/classroom.jpg", label: "Classroom Learning" },
              { src: "/images/applications/medical.jpg", label: "Healthcare" },
              { src: "/images/applications/communication.jpg", label: "Family Communication" },
              { src: "/images/applications/interpreter.jpg", label: "Professional Training" },
            ].map((img) => (
              <div key={img.label} className="relative rounded-lg overflow-hidden">
                <img src={img.src} alt={img.label} className="w-full h-full object-cover" />
                <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/70 to-transparent px-2 py-1">
                  <span className="text-white text-[10px] font-bold">{img.label}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Stats column */}
          <div className="space-y-2">
            {[
              { v: "3.2M", l: "Parameters" },
              { v: "<5ms", l: "Inference" },
              { v: "5,565", l: "Signs Trained" },
              { v: "+225%", l: "Few-shot Gain" },
              { v: "4", l: "Datasets" },
            ].map((s) => (
              <div key={s.l} className="bg-white/90 backdrop-blur rounded-lg px-3 py-1.5 shadow">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600">{s.l}</span>
                  <span className="text-sm font-bold text-[#2D5A4A]">{s.v}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// PAGE 2: PROBLEM
// ============================================================================
function ProblemPage() {
  return (
    <div className="p-5 space-y-4">
      <div className="text-center">
        <h1 className="text-3xl font-serif text-gray-900">The Problem</h1>
        <p className="text-gray-600 mt-1 text-sm">Why existing ASL learning tools fall short</p>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-5 gap-4">
        {/* Left: Issues - 3 cols */}
        <div className="col-span-3 space-y-3">
          <div className="bg-red-50 border border-red-200 rounded-xl p-4">
            <h3 className="font-bold text-red-700 mb-2 text-sm">Traditional Learning Methods</h3>
            <div className="space-y-1.5">
              {[
                { t: "In-person classes", i: "Expensive ($50-100/hr), limited availability, scheduling constraints" },
                { t: "Video tutorials", i: "No feedback mechanism — passive learning only, can't correct mistakes" },
                { t: "Practice partners", i: "Inconsistent feedback quality, hard to find, limited availability" },
              ].map((item) => (
                <div key={item.t} className="bg-white rounded-lg p-2.5">
                  <div className="font-semibold text-gray-900 text-sm">{item.t}</div>
                  <div className="text-red-600 text-xs mt-0.5">{item.i}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-red-50 border border-red-200 rounded-xl p-4">
            <h3 className="font-bold text-red-700 mb-2 text-sm">Current AI Approaches</h3>
            <div className="space-y-1.5">
              {[
                { t: "Video-based recognition", i: "Only ~63% accuracy (skeleton-only), stores raw video (privacy concern)" },
                { t: "Holistic classification", i: "Treats signs as atomic units, cannot analyze components" },
                { t: "Binary feedback only", i: "'Correct' or 'Incorrect' — no guidance on what to fix" },
              ].map((item) => (
                <div key={item.t} className="bg-white rounded-lg p-2.5">
                  <div className="font-semibold text-gray-900 text-sm">{item.t}</div>
                  <div className="text-red-600 text-xs mt-0.5">{item.i}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Images - 2 cols */}
        <div className="col-span-2 space-y-3">
          <div className="rounded-xl overflow-hidden border border-gray-200 bg-white">
            <img src="/graphics/feedback-comparison.svg" alt="Binary vs Component Feedback" className="w-full object-contain" />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-xl overflow-hidden border border-gray-200">
              <img src="/images/asl-learning.jpg" alt="People learning ASL" className="w-full h-32 object-cover" />
            </div>
            <div className="rounded-xl overflow-hidden border border-gray-200">
              <img src="/images/asl-examples.png" alt="ASL Sign Examples" className="w-full h-32 object-cover" />
            </div>
          </div>
          <div className="rounded-xl overflow-hidden border border-gray-200">
            <img src="/images/applications/learning.jpg" alt="Children learning sign language" className="w-full h-32 object-cover" />
          </div>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-white rounded-xl shadow overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-4 py-2 text-left font-bold text-gray-800">Capability</th>
              <th className="px-4 py-2 text-center text-gray-600">In-Person</th>
              <th className="px-4 py-2 text-center text-gray-600">Video Apps</th>
              <th className="px-4 py-2 text-center text-gray-600">Academic AI</th>
              <th className="px-4 py-2 text-center font-bold text-[#2D5A4A]">SignSense</th>
            </tr>
          </thead>
          <tbody>
            {[
              { c: "Recognition accuracy", a: "N/A", b: "N/A", d: "~63%", e: "88.4%" },
              { c: "Component-level feedback", a: "Human only", b: "None", d: "None", e: "4 components" },
              { c: "Error type diagnosis", a: "Subjective", b: "None", d: "None", e: "16 error types" },
              { c: "Privacy preservation", a: "In-person", b: "Video stored", d: "Video stored", e: "Skeleton only" },
              { c: "24/7 availability", a: "No", b: "Yes", d: "Research only", e: "Yes" },
            ].map((r) => (
              <tr key={r.c} className="border-t border-gray-100">
                <td className="px-4 py-1.5 font-medium text-gray-800">{r.c}</td>
                <td className="px-4 py-1.5 text-center text-gray-500">{r.a}</td>
                <td className="px-4 py-1.5 text-center text-gray-500">{r.b}</td>
                <td className="px-4 py-1.5 text-center text-gray-500">{r.d}</td>
                <td className="px-4 py-1.5 text-center font-bold text-[#2D5A4A]">{r.e}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ============================================================================
// PAGE 3: SOLUTION
// ============================================================================
function SolutionPage() {
  return (
    <div className="p-5 space-y-4">
      <div className="text-center">
        <h1 className="text-3xl font-serif text-gray-900">Our Solution</h1>
        <p className="text-gray-600 mt-1 text-sm">Linguistically-informed neural architecture</p>
      </div>

      {/* Phonological Decomposition - full width */}
      <div className="bg-[#2D5A4A] text-white rounded-xl p-4">
        <h2 className="font-bold text-base mb-1.5">Key Innovation: Phonological Decomposition</h2>
        <p className="text-sm opacity-90 leading-relaxed">
          ASL signs are composed of four simultaneous phonological components (Stokoe, 1960).
          SignSense embeds this linguistic theory directly into the neural network, enabling component-level analysis and targeted feedback.
        </p>
        <div className="grid grid-cols-4 gap-3 mt-3">
          {[
            { name: "Handshape", desc: "Finger configuration and grip type" },
            { name: "Location", desc: "Position relative to body" },
            { name: "Movement", desc: "Motion path and dynamics" },
            { name: "Orientation", desc: "Palm and finger direction" },
          ].map((c) => (
            <div key={c.name} className="bg-white/20 rounded-lg p-2.5 text-center">
              <div className="font-bold text-sm">{c.name}</div>
              <div className="text-xs opacity-80">{c.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Architecture + Privacy side by side */}
      <div className="grid grid-cols-3 gap-4">
        {/* Architecture - 2 columns, larger */}
        <div className="col-span-2 bg-white rounded-xl p-5 shadow">
          <h3 className="font-bold text-gray-900 text-base mb-3">PhonSSM Architecture (3.2M Parameters)</h3>

          {/* Flow diagram - all blocks same neutral color */}
          <div className="flex items-center justify-center mb-4">
            {[
              { name: "Input", sub: "30×75×3" },
              { name: "AGAN", sub: "773K" },
              { name: "PDM", sub: "135K" },
              { name: "BiSSM", sub: "1.5M" },
              { name: "HPC", sub: "789K" },
            ].map((s, i) => (
              <div key={s.name} className="flex items-center">
                <div className="rounded-lg px-3 py-2 text-center bg-[#2D5A4A] text-white">
                  <div className="font-bold text-sm">{s.name}</div>
                  <div className="text-[10px] opacity-80">{s.sub}</div>
                </div>
                {i < 4 && (
                  <svg className="w-8 h-6 text-[#2D5A4A] shrink-0" viewBox="0 0 32 24" fill="none">
                    <path d="M2 12 L24 12 M20 7 L26 12 L20 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                )}
              </div>
            ))}
          </div>

          {/* Supporting models - color-coded to match pipeline */}
          <div className="grid grid-cols-3 gap-2 mb-4">
            {[
              { name: "Error Diagnosis Network", params: "500K", desc: "16 error types across all components", color: "#C75D4D", bg: "bg-[#C75D4D]/10", text: "text-[#C75D4D]" },
              { name: "Movement Analyzer", params: "100K", desc: "Speed, smoothness, completeness", color: "#E8B86D", bg: "bg-[#E8B86D]/15", text: "text-[#B8860B]" },
              { name: "Feedback Ranker", params: "10K", desc: "Prioritizes corrections for learning", color: "#7C5CBF", bg: "bg-[#7C5CBF]/10", text: "text-[#7C5CBF]" },
            ].map((m) => (
              <div key={m.name} className={`${m.bg} rounded-lg p-2.5 border-l-3`} style={{ borderLeftWidth: "3px", borderLeftColor: m.color }}>
                <div className="flex justify-between items-start mb-1">
                  <span className="font-bold text-xs text-gray-900">{m.name}</span>
                  <span className={`text-[9px] font-mono ${m.text} px-1 py-0.5 rounded bg-white/60`}>{m.params}</span>
                </div>
                <p className="text-xs text-gray-600">{m.desc}</p>
              </div>
            ))}
          </div>

          {/* Pipeline figure embedded */}
          <div className="border-t border-gray-200 pt-3">
            <h4 className="font-bold text-sm text-gray-700 text-center mb-1">Four-Model Diagnostic Pipeline</h4>
            <p className="text-[10px] text-gray-500 text-center mb-2">From webcam to actionable feedback in {"<"}5ms</p>
            <img src="/graphics/pipeline.svg" alt="Complete Pipeline" className="w-full max-h-[260px] object-contain" />
          </div>
        </div>

        {/* Privacy First - 1 column, matched height */}
        <div className="bg-[#2D5A4A] text-white rounded-xl p-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" strokeWidth="2"/></svg>
              <h2 className="font-bold text-lg">Privacy First</h2>
            </div>
            <p className="text-sm opacity-90 leading-relaxed mb-3">
              SignSense processes <strong>skeleton data only</strong> — your video never leaves your device.
              No face, no background, no identifiable information.
            </p>
          </div>
          <div className="space-y-2">
            <div className="bg-white/20 rounded-lg p-3">
              <div className="text-xs opacity-80 mb-1 font-bold">What we process:</div>
              <div className="font-mono text-sm">(x,y,z) × 21 landmarks</div>
              <div className="font-mono text-sm">× 30 frames per second</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-xs opacity-80 mb-1 font-bold">NOT processed:</div>
              <div className="text-sm">Face, background, skin, clothing, video</div>
            </div>
            <div className="bg-[#E8B86D]/20 rounded-lg p-3">
              <div className="text-xs font-bold text-[#E8B86D] mb-1">Key Benefits</div>
              <div className="text-sm space-y-1">
                <div>• GDPR/HIPAA compliant by design</div>
                <div>• Works offline — no cloud upload</div>
                <div>• Data minimization principle</div>
              </div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-xs opacity-80 mb-1 font-bold">Deployment:</div>
              <div className="text-sm">Web, mobile, and embedded</div>
              <div className="text-sm">CPU-only, {"<"}5ms inference</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// PAGE 4: RESULTS
// ============================================================================
function ResultsPage() {
  return (
    <div className="p-5 space-y-4">
      <div className="text-center">
        <h1 className="text-3xl font-serif text-gray-900">Results</h1>
        <p className="text-gray-600 mt-1 text-sm">State-of-the-art performance on WLASL benchmarks</p>
      </div>

      {/* Benchmark Chart - prominent */}
      <div className="rounded-xl overflow-hidden border border-gray-200 bg-white">
        <img src="/graphics/benchmarks.svg" alt="WLASL Benchmark Results" className="w-full max-h-[300px] object-contain" />
      </div>

      {/* Results Grid */}
      <div className="grid grid-cols-2 gap-4">
        {/* Demo Video */}
        <div className="rounded-xl overflow-hidden relative h-72">
          <video src="/videos/demo.mp4" autoPlay loop muted playsInline className="w-full h-full object-cover" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent" />
          <div className="absolute bottom-3 left-3 right-3">
            <div className="text-white font-bold text-sm">Live Demo: Real-time Feedback</div>
            <div className="text-white/80 text-xs">30 FPS • {"<"}5ms latency • CPU only</div>
          </div>
        </div>

        {/* Detailed Results */}
        <div className="space-y-3">
          {/* Benchmark Table */}
          <div className="bg-white rounded-xl shadow overflow-hidden">
            <div className="bg-[#2D5A4A] text-white px-3 py-1.5">
              <h3 className="font-bold text-xs">Top-1 Accuracy vs Previous Best (Skeleton-Only)</h3>
            </div>
            <table className="w-full text-xs">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-1.5 text-left">Dataset</th>
                  <th className="px-3 py-1.5 text-right">Previous</th>
                  <th className="px-3 py-1.5 text-right">Ours</th>
                  <th className="px-3 py-1.5 text-right">Gain</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { d: "WLASL100", p: "63.2%", o: "88.4%", g: "+25.2%" },
                  { d: "WLASL300", p: "58.4%", o: "74.4%", g: "+16.0%" },
                  { d: "WLASL1000", p: "47.1%", o: "62.9%", g: "+15.8%" },
                  { d: "WLASL2000", p: "53.7%", o: "72.1%", g: "+18.4%" },
                ].map((r) => (
                  <tr key={r.d} className="border-t border-gray-100">
                    <td className="px-3 py-1 font-medium">{r.d}</td>
                    <td className="px-3 py-1 text-right text-gray-500">{r.p}</td>
                    <td className="px-3 py-1 text-right font-bold text-[#2D5A4A]">{r.o}</td>
                    <td className="px-3 py-1 text-right text-green-600 font-medium">{r.g}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Ablation */}
          <div className="bg-gray-50 rounded-xl p-3">
            <h3 className="font-bold text-xs text-gray-900 mb-1.5">Ablation Study</h3>
            <div className="space-y-1 text-xs">
              {[
                { c: "Full PhonSSM", a: "88.4%", d: "—", hl: true },
                { c: "− PDM (phonological)", a: "76.2%", d: "−12.2%" },
                { c: "− BiSSM (temporal)", a: "79.8%", d: "−8.6%" },
                { c: "− AGAN (graph)", a: "81.1%", d: "−7.3%" },
              ].map((r) => (
                <div key={r.c} className={`flex justify-between px-2 py-0.5 rounded ${r.hl ? 'bg-[#2D5A4A]/10 font-bold' : ''}`}>
                  <span>{r.c}</span>
                  <span className={r.hl ? 'text-[#2D5A4A]' : 'text-red-600'}>{r.a} {r.d !== '—' && `(${r.d})`}</span>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-gray-500 mt-1 italic">PDM contributes most — linguistic theory is the key innovation</p>
          </div>
        </div>
      </div>

      {/* Training Data + Key Stats */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-[#2D5A4A]/10 rounded-xl p-3">
          <h3 className="font-bold text-sm text-[#2D5A4A] mb-2">Training Data (4 Datasets)</h3>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between"><span className="font-medium">WLASL</span><span className="text-gray-600">2,000 signs • 21K samples</span></div>
            <div className="flex justify-between"><span className="font-medium">ASL Citizen</span><span className="text-gray-600">2,731 signs • 83K samples</span></div>
            <div className="flex justify-between"><span className="font-medium">How2Sign</span><span className="text-gray-600">Augmentation data</span></div>
            <div className="flex justify-between"><span className="font-medium">YouTube-ASL</span><span className="text-gray-600">Pretraining data</span></div>
          </div>
        </div>
        <div className="bg-white rounded-xl p-3 shadow">
          <h3 className="font-bold text-sm text-gray-900 mb-2">Key Performance Metrics</h3>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between bg-gray-50 rounded-lg px-2 py-1.5"><span className="text-gray-600">Avg Improvement</span><span className="font-bold text-[#2D5A4A]">+18.9%</span></div>
            <div className="flex justify-between bg-gray-50 rounded-lg px-2 py-1.5"><span className="text-gray-600">Few-Shot Gain</span><span className="font-bold text-[#2D5A4A]">+225%</span></div>
            <div className="flex justify-between bg-gray-50 rounded-lg px-2 py-1.5"><span className="text-gray-600">Inference</span><span className="font-bold text-[#2D5A4A]">{"<"}5ms</span></div>
            <div className="flex justify-between bg-gray-50 rounded-lg px-2 py-1.5"><span className="text-gray-600">Parameters</span><span className="font-bold text-[#2D5A4A]">3.2M</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// PAGE 5: FUTURE
// ============================================================================
function FuturePage() {
  return (
    <div className="p-5 space-y-4">
      <div className="text-center">
        <h1 className="text-3xl font-serif text-gray-900">Future Vision</h1>
        <p className="text-gray-600 mt-1 text-sm">From working prototype to humanoid teaching companion</p>
      </div>

      {/* Current Status */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-green-50 border-2 border-green-400 rounded-xl p-4">
          <h3 className="font-bold text-green-800 text-base mb-2 flex items-center gap-2">
            <span className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center text-white text-xs">✓</span>
            Currently Production Ready
          </h3>
          <div className="grid grid-cols-2 gap-1.5 text-sm text-green-800">
            {[
              "PhonSSM Classifier (88.4%)",
              "Error Diagnosis (16 types)",
              "Movement Analyzer (6 classes)",
              "Feedback Ranker (TFLite)",
              "Live Web Application",
              "Real-time inference (<5ms)",
            ].map((item) => (
              <div key={item} className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 bg-green-500 rounded-full shrink-0"></span>
                <span className="text-xs">{item}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-xl p-4 shadow">
          <h3 className="font-bold text-gray-900 text-sm mb-2">Live Demo Specifications</h3>
          <div className="grid grid-cols-3 gap-2">
            {[
              { l: "Frame Rate", v: "30 FPS" },
              { l: "Latency", v: "<5ms" },
              { l: "Vocabulary", v: "5,565 signs" },
              { l: "Error Types", v: "16 categories" },
              { l: "Model Size", v: "3.2M params" },
              { l: "Runtime", v: "CPU only" },
            ].map((s) => (
              <div key={s.l} className="bg-gray-50 rounded-lg p-1.5 text-center">
                <div className="font-bold text-sm text-[#2D5A4A]">{s.v}</div>
                <div className="text-[10px] text-gray-600">{s.l}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="bg-white rounded-xl p-4 shadow">
        <h3 className="font-bold text-gray-900 text-sm mb-3">Development Roadmap</h3>
        <div className="grid grid-cols-4 gap-3">
          {[
            { p: "1", t: "Core AI", s: "done", items: ["PhonSSM architecture", "4-model pipeline", "Web demo"] },
            { p: "2", t: "Mobile", s: "done", items: ["React Native app", "TFLite conversion", "Offline mode"] },
            { p: "3", t: "Scale", s: "current", items: ["10,000+ signs", "Multi-language", "Public API"] },
            { p: "4", t: "Hardware", s: "future", items: ["Humanoid robot", "Articulated hands", "Teaching mode"] },
          ].map((phase) => (
            <div
              key={phase.p}
              className={`rounded-xl p-3 ${
                phase.s === 'done' ? 'bg-green-100 border-2 border-green-400' :
                phase.s === 'current' ? 'bg-[#2D5A4A] text-white' :
                'bg-gray-100'
              }`}
            >
              <div className={`text-xs font-bold ${phase.s === 'current' ? 'text-[#E8B86D]' : phase.s === 'done' ? 'text-green-600' : 'text-gray-500'}`}>
                Phase {phase.p}
              </div>
              <div className={`text-base font-bold ${phase.s === 'current' ? 'text-white' : 'text-gray-900'}`}>{phase.t}</div>
              <div className={`text-[11px] mt-1 space-y-0.5 ${phase.s === 'current' ? 'text-white/80' : 'text-gray-600'}`}>
                {phase.items.map((item) => <div key={item}>• {item}</div>)}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Robot Vision - larger images, compact outline */}
      <div className="bg-gradient-to-r from-[#2D5A4A] to-[#3D7A6A] text-white rounded-xl p-4">
        <h3 className="text-lg font-bold mb-2">SignSense Companion Robot</h3>
        <div className="grid grid-cols-5 gap-4">
          {/* Description - 2 cols */}
          <div className="col-span-2">
            <p className="text-sm opacity-90 leading-relaxed mb-3">
              Future vision: A humanoid robot with fully articulated hands capable of demonstrating ASL signs
              and providing real-time teaching feedback through natural interaction.
            </p>
            <div className="grid grid-cols-2 gap-1.5 mb-3">
              {[
                { f: "21 DOF per hand", d: "Full finger articulation" },
                { f: "Depth camera", d: "3D hand tracking" },
                { f: "Voice + sign", d: "Multimodal output" },
                { f: "Adaptive pace", d: "Personalized learning" },
              ].map((item) => (
                <div key={item.f} className="flex items-start gap-1.5 text-sm bg-white/10 rounded-lg p-1.5">
                  <span className="w-3.5 h-3.5 bg-[#E8B86D] rounded flex items-center justify-center text-[#2D5A4A] text-[9px] shrink-0 mt-0.5">✓</span>
                  <div>
                    <div className="font-bold text-[11px]">{item.f}</div>
                    <div className="text-[9px] opacity-80">{item.d}</div>
                  </div>
                </div>
              ))}
            </div>
            <p className="text-[11px] opacity-80 leading-relaxed">
              The robot integrates our PhonSSM architecture for real-time sign recognition and uses the
              error diagnosis network to identify exactly which phonological components need correction.
              When a student makes an error — such as incorrect handshape or movement path — the robot
              physically demonstrates the correct form using its 21-DOF articulated hands while simultaneously
              providing verbal guidance. This multimodal feedback loop mirrors how human ASL tutors teach, but with the consistency and patience
              of an AI system available 24/7. The same lightweight 3.2M parameter model runs on-device, ensuring real-time responsiveness and full privacy.
            </p>
          </div>

          {/* Robot concept - 1 col */}
          <div className="flex flex-col">
            <div className="bg-white/10 rounded-xl p-2 flex items-center justify-center flex-1">
              <img src="/graphics/robot-concept.svg" alt="Robot Concept" className="w-full h-full object-contain" />
            </div>
            <div className="text-[9px] opacity-70 mt-1 text-center leading-snug">
              <span className="font-bold">Prototype: Teaching Companion</span><br/>
              Integrated depth camera, stereo speakers, 7" touchscreen display. Runs PhonSSM on embedded ARM CPU with {"<"}10ms inference. Demonstrates signs at adjustable speed for guided practice.
            </div>
          </div>

          {/* Robotic hand SVG - 1 col */}
          <div className="flex flex-col">
            <div className="bg-white/10 rounded-xl p-2 flex items-center justify-center flex-1">
              <img src="/graphics/robotic-hand.svg" alt="Robotic Hand Design" className="w-full h-full object-contain" />
            </div>
            <div className="text-[9px] opacity-70 mt-1 text-center leading-snug">
              <span className="font-bold">Prototype: 21-DOF Articulated Hand</span><br/>
              Each finger has 4 independent joints driven by micro-servos. Tendon-actuated linkage system enables human-like range of motion for all 80+ ASL handshapes.
            </div>
          </div>

          {/* Kim et al. hand - 1 col */}
          <div className="flex flex-col">
            <div className="bg-white/10 rounded-xl p-2 flex items-center justify-center flex-1">
              <img src="/images/robotic-hand-kim2021.png" alt="Kim et al. 2021 Hand" className="w-full h-full object-contain rounded" />
            </div>
            <div className="text-[9px] opacity-70 mt-1 text-center leading-snug">
              <span className="font-bold">Reference: Dexterous Anthropomorphic Hand</span><br/>
              Kim et al. (2021) integrated linkage-driven design, <em>Nature Communications</em> 12, 7177. Demonstrates feasibility of human-level dexterity for sign production.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// MAIN
// ============================================================================
export default function ToshibaChallengePage() {
  const [activeTab, setActiveTab] = useState("overview");

  const renderContent = () => {
    switch (activeTab) {
      case "overview": return <OverviewPage />;
      case "problem": return <ProblemPage />;
      case "solution": return <SolutionPage />;
      case "results": return <ResultsPage />;
      case "future": return <FuturePage />;
      default: return <OverviewPage />;
    }
  };

  return (
    <main className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-3">
          <div className="flex items-center justify-between mb-2">
            <img src="/logo.svg" alt="SignSense" className="h-8" />
            <span className="text-sm text-gray-500 font-medium">Toshiba Challenge 2026</span>
          </div>
          <nav className="flex gap-2">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? "bg-[#2D5A4A] text-white"
                    : "text-gray-600 hover:bg-gray-100"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Content */}
      <div className="max-w-6xl mx-auto">
        {renderContent()}
      </div>
    </main>
  );
}
