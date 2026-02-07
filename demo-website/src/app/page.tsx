"use client";

import { motion, useScroll, useTransform, useInView } from "framer-motion";
import { useRef, useEffect, useState } from "react";

// Animation variants
const fadeUpVariants = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: [0.25, 0.1, 0.25, 1] } }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.15, delayChildren: 0.2 }
  }
};

const scaleUpVariants = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.6, ease: "easeOut" } }
};

// Counter component for animated numbers
function AnimatedCounter({ target, suffix = "", duration = 2 }: { target: number; suffix?: string; duration?: number }) {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });

  useEffect(() => {
    if (isInView) {
      let start = 0;
      const increment = target / (duration * 60);
      const timer = setInterval(() => {
        start += increment;
        if (start >= target) {
          setCount(target);
          clearInterval(timer);
        } else {
          setCount(Math.floor(start * 10) / 10);
        }
      }, 1000 / 60);
      return () => clearInterval(timer);
    }
  }, [isInView, target, duration]);

  return <span ref={ref}>{count.toFixed(1)}{suffix}</span>;
}

// Section wrapper with scroll animations
function Section({ children, className = "", id = "" }: { children: React.ReactNode; className?: string; id?: string }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <motion.section
      ref={ref}
      id={id}
      className={className}
      initial="hidden"
      animate={isInView ? "visible" : "hidden"}
      variants={staggerContainer}
    >
      {children}
    </motion.section>
  );
}

// Navigation
function Navigation() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <motion.nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? "bg-bg-primary/90 backdrop-blur-lg shadow-sm" : ""
      }`}
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
        <motion.div
          className="flex items-center gap-2"
          whileHover={{ scale: 1.02 }}
        >
          <img src="/logo.svg" alt="SignSense" className="h-10" />
        </motion.div>
        <div className="hidden md:flex items-center gap-8">
          {["Technology", "Results", "Applications", "Demo"].map((item) => (
            <a
              key={item}
              href={`#${item.toLowerCase()}`}
              className="text-text-secondary hover:text-accent-primary transition-colors animated-underline"
            >
              {item}
            </a>
          ))}
          <motion.a
            href="#demo"
            className="bg-accent-gradient text-white px-6 py-2.5 rounded-full font-medium"
            whileHover={{ scale: 1.05, boxShadow: "0 10px 30px rgba(45, 90, 74, 0.3)" }}
            whileTap={{ scale: 0.98 }}
          >
            Watch Demo
          </motion.a>
        </div>
      </div>
    </motion.nav>
  );
}

// Hero Section
function HeroSection() {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const y = useTransform(scrollYProgress, [0, 1], [0, 150]);
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);

  return (
    <section ref={ref} className="min-h-screen relative overflow-hidden hero-pattern">
      <motion.div
        className="max-w-7xl mx-auto px-6 pt-32 pb-20 min-h-screen flex flex-col justify-center"
        style={{ y, opacity }}
      >
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <div className="space-y-8">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <span className="inline-block px-4 py-2 bg-accent-primary/10 text-accent-primary rounded-full text-sm font-medium mb-6">
                Toshiba Challenge 2026
              </span>
            </motion.div>

            <motion.h1
              className="font-display text-hero text-text-primary leading-tight"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.1 }}
            >
              Learn Sign Language with{" "}
              <span className="text-gradient">AI That Actually Understands</span>
            </motion.h1>

            <motion.p
              className="text-xl text-text-secondary max-w-xl leading-relaxed"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              SignSense uses <strong>four specialized neural networks</strong> working together
              to give you real-time, component-specific feedback on your signing technique.
              Not just "right" or "wrong" — but exactly what to fix.
            </motion.p>

            <motion.div
              className="flex flex-wrap gap-4 pt-4"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
            >
              <motion.a
                href="#demo"
                className="inline-flex items-center gap-2 bg-accent-gradient text-white px-8 py-4 rounded-full font-semibold text-lg"
                whileHover={{ scale: 1.05, boxShadow: "0 20px 40px rgba(45, 90, 74, 0.3)" }}
                whileTap={{ scale: 0.98 }}
              >
                <span>Watch Demo</span>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </motion.a>
              <motion.a
                href="#technology"
                className="inline-flex items-center gap-2 border-2 border-accent-primary text-accent-primary px-8 py-4 rounded-full font-semibold text-lg"
                whileHover={{ backgroundColor: "rgba(45, 90, 74, 0.1)" }}
                whileTap={{ scale: 0.98 }}
              >
                Explore Technology
              </motion.a>
            </motion.div>

            {/* Quick Stats */}
            <motion.div
              className="flex gap-8 pt-8"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.5 }}
            >
              {[
                { value: "88.4", label: "% Accuracy" },
                { value: "4", label: "AI Models" },
                { value: "5,565", label: "Signs Supported" },
              ].map((stat, i) => (
                <div key={i} className="text-center">
                  <div className="text-3xl font-bold text-accent-primary">{stat.value}</div>
                  <div className="text-sm text-text-tertiary">{stat.label}</div>
                </div>
              ))}
            </motion.div>
          </div>

          {/* Right Visual - Hand Animation */}
          <motion.div
            className="relative"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.4 }}
          >
            <div className="relative w-full aspect-square max-w-lg mx-auto">
              {/* Glowing background */}
              <div className="absolute inset-0 bg-gradient-radial from-accent-primary/20 via-accent-secondary/10 to-transparent rounded-full animate-pulse-soft" />

              {/* Skeleton hand illustration */}
              <motion.div
                className="absolute inset-0 flex items-center justify-center"
                animate={{ rotate: [0, 5, -5, 0] }}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              >
                <svg viewBox="0 0 200 280" className="w-3/4 h-3/4">
                  {/* Palm */}
                  <motion.ellipse
                    cx="100" cy="180" rx="50" ry="60"
                    fill="none"
                    stroke="#2D5A4A"
                    strokeWidth="3"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 2, delay: 0.5 }}
                  />

                  {/* Fingers */}
                  {[
                    { x1: 60, y1: 130, x2: 45, y2: 50 },  // Thumb
                    { x1: 75, y1: 120, x2: 70, y2: 20 },  // Index
                    { x1: 100, y1: 115, x2: 100, y2: 10 }, // Middle
                    { x1: 125, y1: 120, x2: 130, y2: 25 }, // Ring
                    { x1: 145, y1: 135, x2: 155, y2: 55 }, // Pinky
                  ].map((finger, i) => (
                    <motion.line
                      key={i}
                      x1={finger.x1} y1={finger.y1}
                      x2={finger.x2} y2={finger.y2}
                      stroke="#2D5A4A"
                      strokeWidth="3"
                      strokeLinecap="round"
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 1, delay: 0.8 + i * 0.15 }}
                    />
                  ))}

                  {/* Joint nodes */}
                  {[
                    { cx: 45, cy: 50 }, { cx: 52, cy: 90 },
                    { cx: 70, cy: 20 }, { cx: 72, cy: 70 },
                    { cx: 100, cy: 10 }, { cx: 100, cy: 60 },
                    { cx: 130, cy: 25 }, { cx: 127, cy: 72 },
                    { cx: 155, cy: 55 }, { cx: 150, cy: 95 },
                  ].map((joint, i) => (
                    <motion.circle
                      key={i}
                      cx={joint.cx} cy={joint.cy} r="6"
                      fill="#E8B86D"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ duration: 0.3, delay: 1.5 + i * 0.05 }}
                    />
                  ))}
                </svg>
              </motion.div>

              {/* Floating labels */}
              {[
                { label: "Handshape", x: "10%", y: "20%", delay: 2 },
                { label: "Location", x: "70%", y: "15%", delay: 2.2 },
                { label: "Movement", x: "80%", y: "60%", delay: 2.4 },
                { label: "Orientation", x: "5%", y: "70%", delay: 2.6 },
              ].map((item, i) => (
                <motion.div
                  key={i}
                  className="absolute bg-white/90 backdrop-blur-sm px-3 py-1.5 rounded-full text-sm font-medium text-accent-primary shadow-lg"
                  style={{ left: item.x, top: item.y }}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: item.delay }}
                >
                  {item.label}
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Scroll indicator */}
        <motion.div
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1, y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity, delay: 2 }}
        >
          <div className="flex flex-col items-center text-text-tertiary">
            <span className="text-sm mb-2">Scroll to explore</span>
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </div>
        </motion.div>
      </motion.div>
    </section>
  );
}

// Problem Section
function ProblemSection() {
  return (
    <Section className="py-32 bg-bg-secondary" id="problem">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          <div className="space-y-8">
            <motion.span
              variants={fadeUpVariants}
              className="inline-block text-accent-primary font-semibold tracking-widest text-sm uppercase"
            >
              The Challenge
            </motion.span>

            <motion.h2
              variants={fadeUpVariants}
              className="font-display text-h1 text-text-primary"
            >
              70 Million People Use Sign Language.{" "}
              <span className="text-accent-tertiary">Learning It Shouldn't Be This Hard.</span>
            </motion.h2>

            <motion.div variants={fadeUpVariants} className="space-y-6">
              {[
                { icon: "🎓", title: "Limited Access", desc: "Qualified ASL instructors are scarce, especially outside urban areas" },
                { icon: "🔇", title: "No Real-Time Feedback", desc: "Books and videos can't tell you if you're signing correctly" },
                { icon: "❌", title: "Binary Assessment", desc: "Traditional apps only say 'right' or 'wrong' — not what to fix" },
              ].map((item, i) => (
                <motion.div
                  key={i}
                  className="flex gap-4 p-4 bg-white rounded-xl shadow-sm"
                  variants={fadeUpVariants}
                  whileHover={{ x: 10, boxShadow: "0 10px 30px rgba(0,0,0,0.1)" }}
                >
                  <span className="text-3xl">{item.icon}</span>
                  <div>
                    <h4 className="font-semibold text-text-primary">{item.title}</h4>
                    <p className="text-text-secondary">{item.desc}</p>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>

          {/* Comparison Visual */}
          <motion.div variants={scaleUpVariants} className="relative">
            <div className="bg-white rounded-2xl shadow-xl p-8 space-y-6">
              <h3 className="font-display text-2xl text-center text-text-primary mb-8">
                The SignSense Difference
              </h3>

              <div className="grid grid-cols-2 gap-6">
                {/* Traditional */}
                <div className="space-y-4">
                  <div className="text-center p-4 bg-red-50 rounded-xl">
                    <span className="text-4xl">📱</span>
                    <h4 className="font-semibold mt-2 text-text-primary">Traditional Apps</h4>
                  </div>
                  <div className="p-4 bg-red-100/50 rounded-lg text-center">
                    <span className="text-4xl text-error">✗</span>
                    <p className="text-error font-semibold mt-2">Wrong</p>
                    <p className="text-sm text-text-secondary mt-1">No explanation why</p>
                  </div>
                </div>

                {/* SignSense */}
                <div className="space-y-4">
                  <div className="text-center p-4 bg-accent-primary/10 rounded-xl">
                    <span className="text-2xl font-bold text-accent-primary">SignSense</span>
                  </div>
                  <div className="p-4 bg-accent-primary/5 rounded-lg">
                    <p className="text-success font-bold text-lg">Correct!</p>
                    <div className="mt-3 space-y-2 text-sm">
                      {[
                        { label: "Handshape", value: 94 },
                        { label: "Location", value: 87 },
                        { label: "Movement", value: 91 },
                      ].map((item, i) => (
                        <div key={i} className="flex items-center gap-2">
                          <span className="text-text-secondary w-20">{item.label}</span>
                          <div className="flex-1 bg-bg-tertiary rounded-full h-2">
                            <motion.div
                              className="bg-accent-primary h-full rounded-full"
                              initial={{ width: 0 }}
                              whileInView={{ width: `${item.value}%` }}
                              transition={{ duration: 1, delay: i * 0.2 }}
                            />
                          </div>
                          <span className="text-accent-primary font-medium w-10">{item.value}%</span>
                        </div>
                      ))}
                    </div>
                    <p className="mt-3 text-xs text-accent-primary bg-accent-primary/10 p-2 rounded font-medium">
                      Tip: Extend index finger more
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </Section>
  );
}

// Solution Overview
function SolutionSection() {
  const models = [
    {
      num: "1",
      title: "RECOGNIZE",
      subtitle: "PhonSSM Classifier",
      desc: "Identifies your sign with 88.4% accuracy using phonological decomposition",
      bgColor: "bg-accent-primary"
    },
    {
      num: "2",
      title: "DIAGNOSE",
      subtitle: "Error Network",
      desc: "Pinpoints 16 specific error types across handshape, location, movement & orientation",
      bgColor: "bg-accent-tertiary"
    },
    {
      num: "3",
      title: "ANALYZE",
      subtitle: "Movement Model",
      desc: "Assesses speed, smoothness, and completeness of your signing motion",
      bgColor: "bg-accent-secondary"
    },
    {
      num: "4",
      title: "PRIORITIZE",
      subtitle: "Feedback Ranker",
      desc: "Orders corrections by importance so you focus on what matters most",
      bgColor: "bg-[#7C5CBF]"
    },
  ];

  return (
    <Section className="py-32 bg-bg-primary" id="technology">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center max-w-3xl mx-auto mb-20">
          <motion.span
            variants={fadeUpVariants}
            className="inline-block text-accent-primary font-semibold tracking-widest text-sm uppercase mb-4"
          >
            Our Approach
          </motion.span>
          <motion.h2
            variants={fadeUpVariants}
            className="font-display text-h1 text-text-primary mb-6"
          >
            Four Specialized Models.{" "}
            <span className="text-gradient">One Seamless Experience.</span>
          </motion.h2>
          <motion.p
            variants={fadeUpVariants}
            className="text-xl text-text-secondary"
          >
            Unlike simple classifiers, SignSense employs a diagnostic pipeline that identifies
            exactly what you need to fix — not just that something is wrong.
          </motion.p>
        </div>

        {/* Model cards */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          {models.map((model, i) => (
            <motion.div
              key={i}
              variants={fadeUpVariants}
              className="bg-white rounded-2xl p-6 shadow-sm card-hover border border-transparent hover:border-accent-primary/20"
              custom={i}
            >
              <motion.div
                className={`w-14 h-14 ${model.bgColor} rounded-xl flex items-center justify-center mb-4`}
                whileHover={{ scale: 1.1, rotate: 5 }}
                transition={{ type: "spring", stiffness: 400 }}
              >
                <span className="text-white text-2xl font-bold">{model.num}</span>
              </motion.div>
              <h3 className="text-sm font-bold tracking-wider text-accent-primary mb-1">
                {model.title}
              </h3>
              <h4 className="font-display text-xl text-text-primary mb-3">
                {model.subtitle}
              </h4>
              <p className="text-text-secondary text-sm leading-relaxed">
                {model.desc}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Pipeline SVG Graphic */}
        <motion.div
          variants={scaleUpVariants}
          className="rounded-2xl overflow-hidden shadow-lg"
        >
          <img
            src="/graphics/pipeline.svg"
            alt="SignSense Four-Model Pipeline Architecture"
            className="w-full"
          />
        </motion.div>

        {/* Feedback Comparison */}
        <motion.div
          variants={fadeUpVariants}
          className="mt-20"
        >
          <img
            src="/graphics/feedback-comparison.svg"
            alt="Traditional Apps vs SignSense Feedback Comparison"
            className="w-full rounded-2xl shadow-lg"
          />
        </motion.div>
      </div>
    </Section>
  );
}

// Architecture Deep Dive
function ArchitectureSection() {
  const components = [
    {
      name: "AGAN",
      full: "Anatomical Graph Attention Network",
      desc: "Treats your skeleton as a graph, understanding that fingers connect to wrists and hands have specific topology",
      params: "773K params",
      color: "#2D5A4A"
    },
    {
      name: "PDM",
      full: "Phonological Disentanglement Module",
      desc: "Separates features into four linguistic components — handshape, location, movement, orientation",
      params: "135K params",
      color: "#E8B86D"
    },
    {
      name: "BiSSM",
      full: "Bidirectional State Space Model",
      desc: "Captures temporal patterns with O(n) efficiency, understanding how your sign unfolds over time",
      params: "1.5M params",
      color: "#C75D4D"
    },
    {
      name: "HPC",
      full: "Hierarchical Prototypical Classifier",
      desc: "Matches your signing to learned prototypes, excelling at rare signs with few training examples",
      params: "789K params",
      color: "#3D8B6E"
    },
  ];

  return (
    <Section className="py-32 bg-bg-secondary" id="architecture">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center max-w-3xl mx-auto mb-20">
          <motion.span
            variants={fadeUpVariants}
            className="inline-block text-accent-primary font-semibold tracking-widest text-sm uppercase mb-4"
          >
            Core Technology
          </motion.span>
          <motion.h2
            variants={fadeUpVariants}
            className="font-display text-h1 text-text-primary mb-6"
          >
            Built on 60 Years of{" "}
            <span className="text-gradient">Sign Language Linguistics</span>
          </motion.h2>
          <motion.p
            variants={fadeUpVariants}
            className="text-xl text-text-secondary"
          >
            PhonSSM's architecture embeds Stokoe's phonological theory directly into the neural network,
            enabling unprecedented accuracy and interpretable feedback.
          </motion.p>
        </div>

        {/* Architecture diagram */}
        <motion.div
          variants={scaleUpVariants}
          className="bg-white rounded-3xl shadow-xl p-8 md:p-12 max-w-4xl mx-auto"
        >
          <div className="space-y-4">
            {/* Input */}
            <motion.div
              className="text-center p-4 bg-bg-secondary rounded-xl"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <span className="text-2xl">📹</span>
              <p className="font-medium text-text-primary mt-2">Input: 30 frames × 75 landmarks × 3 coordinates</p>
            </motion.div>

            {/* Arrow */}
            <div className="flex justify-center">
              <motion.div
                className="w-0.5 h-8 bg-accent-primary"
                initial={{ scaleY: 0 }}
                whileInView={{ scaleY: 1 }}
                transition={{ duration: 0.3, delay: 0.2 }}
              />
            </div>

            {/* Components */}
            {components.map((comp, i) => (
              <div key={i}>
                <motion.div
                  className="architecture-box rounded-xl p-6"
                  style={{ borderLeftColor: comp.color, borderLeftWidth: "4px" }}
                  initial={{ opacity: 0, x: i % 2 === 0 ? -30 : 30 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 + i * 0.15 }}
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="flex justify-between items-start flex-wrap gap-4">
                    <div>
                      <h4 className="font-bold text-accent-primary text-lg">{comp.name}</h4>
                      <p className="text-text-secondary text-sm">{comp.full}</p>
                    </div>
                    <span className="text-xs bg-bg-tertiary px-3 py-1 rounded-full text-text-secondary font-mono">
                      {comp.params}
                    </span>
                  </div>
                  <p className="mt-3 text-text-primary">{comp.desc}</p>
                </motion.div>

                {i < components.length - 1 && (
                  <div className="flex justify-center">
                    <motion.div
                      className="w-0.5 h-6 bg-accent-primary/30"
                      initial={{ scaleY: 0 }}
                      whileInView={{ scaleY: 1 }}
                      transition={{ duration: 0.2, delay: 0.5 + i * 0.1 }}
                    />
                  </div>
                )}
              </div>
            ))}

            {/* Arrow */}
            <div className="flex justify-center">
              <motion.div
                className="w-0.5 h-8 bg-accent-primary"
                initial={{ scaleY: 0 }}
                whileInView={{ scaleY: 1 }}
                transition={{ duration: 0.3, delay: 1 }}
              />
            </div>

            {/* Output */}
            <motion.div
              className="text-center p-4 bg-accent-primary/10 rounded-xl border-2 border-accent-primary"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 1.1 }}
            >
              <span className="text-2xl">✨</span>
              <p className="font-medium text-accent-primary mt-2">Output: Sign + Component Scores + Actionable Feedback</p>
            </motion.div>
          </div>
        </motion.div>

        {/* Key insight callout */}
        <motion.div
          variants={fadeUpVariants}
          className="mt-12 max-w-3xl mx-auto bg-accent-primary text-white rounded-2xl p-10 text-center"
        >
          <p className="text-sm uppercase tracking-widest mb-4 opacity-80">Key Insight</p>
          <p className="text-2xl font-medium leading-relaxed">
            By learning ~135 phonological primitives instead of 5,565 independent patterns,
            PhonSSM achieves <strong className="text-accent-secondary">225% better accuracy</strong> on signs with limited training data.
          </p>
        </motion.div>
      </div>
    </Section>
  );
}

// Results Section
function ResultsSection() {
  const stats = [
    { value: 88.4, label: "WLASL100 Accuracy", suffix: "%" },
    { value: 72.1, label: "WLASL2000 Accuracy", suffix: "%" },
    { value: 25.2, label: "Improvement Over Prior Art", suffix: "%" },
    { value: 225, label: "Few-shot Learning Gain", suffix: "%" },
  ];

  // Animated bar data for background
  const bars = [
    { width: "88%", delay: 0, label: "PhonSSM" },
    { width: "63%", delay: 0.2, label: "Prior Best" },
    { width: "74%", delay: 0.4, label: "WLASL300" },
    { width: "72%", delay: 0.6, label: "WLASL2000" },
  ];

  return (
    <Section className="py-32 bg-bg-primary relative overflow-hidden" id="results">
      {/* Animated Background Bars */}
      <div className="absolute inset-0 opacity-[0.04] pointer-events-none">
        {bars.map((bar, i) => (
          <motion.div
            key={i}
            className="absolute h-24 bg-accent-primary rounded-r-full"
            style={{ top: `${15 + i * 22}%` }}
            initial={{ width: "0%", x: "-100%" }}
            whileInView={{ width: bar.width, x: "0%" }}
            transition={{
              duration: 2,
              delay: bar.delay,
              ease: [0.25, 0.1, 0.25, 1],
            }}
            viewport={{ once: true }}
          />
        ))}
      </div>

      <div className="max-w-7xl mx-auto px-6 relative z-10">
        <div className="text-center max-w-3xl mx-auto mb-20">
          <motion.p
            variants={fadeUpVariants}
            className="text-accent-primary font-semibold tracking-widest text-sm uppercase mb-4"
          >
            Benchmark Results
          </motion.p>
          <motion.h2
            variants={fadeUpVariants}
            className="font-display text-h1 text-text-primary mb-6"
          >
            State-of-the-Art Performance{" "}
            <span className="text-gradient">Across All Benchmarks</span>
          </motion.h2>
          <motion.p
            variants={fadeUpVariants}
            className="text-xl text-text-secondary"
          >
            PhonSSM outperforms all previous skeleton-based methods on WLASL, achieving the highest accuracy ever reported.
          </motion.p>
        </div>

        {/* Big stats with animated bars behind */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-20">
          {stats.map((stat, i) => (
            <motion.div
              key={i}
              variants={fadeUpVariants}
              className="relative bg-white rounded-2xl p-8 text-center shadow-sm card-hover overflow-hidden"
            >
              {/* Mini animated bar inside card */}
              <motion.div
                className="absolute bottom-0 left-0 h-1 bg-accent-primary/20"
                initial={{ width: "0%" }}
                whileInView={{ width: `${stat.value}%` }}
                transition={{ duration: 1.5, delay: i * 0.15, ease: "easeOut" }}
                viewport={{ once: true }}
              />
              <div className="text-5xl md:text-6xl font-bold text-accent-primary stat-number">
                <AnimatedCounter target={stat.value} suffix={stat.suffix} />
              </div>
              <div className="text-text-secondary mt-3 font-medium text-sm">{stat.label}</div>
            </motion.div>
          ))}
        </div>

        {/* Benchmark Chart Graphic */}
        <motion.div
          variants={scaleUpVariants}
          className="rounded-2xl overflow-hidden shadow-lg"
        >
          <img
            src="/graphics/benchmarks.svg"
            alt="WLASL Benchmark Results - PhonSSM vs Previous State-of-the-Art"
            className="w-full"
          />
        </motion.div>

        {/* Skeleton Hand Visualization */}
        <motion.div
          variants={fadeUpVariants}
          className="mt-20 grid md:grid-cols-2 gap-12 items-center"
        >
          <div>
            <h3 className="font-display text-3xl text-text-primary mb-4">
              75 Landmarks Per Frame
            </h3>
            <p className="text-text-secondary text-lg leading-relaxed mb-6">
              MediaPipe extracts precise 3D coordinates from your webcam feed — pose, hands, and fingertips.
              Our models analyze these landmarks in real-time to understand exactly what you're signing.
            </p>
            <ul className="space-y-3">
              <li className="flex items-center gap-3 text-text-secondary">
                <span className="w-2 h-2 bg-accent-primary rounded-full"></span>
                33 pose landmarks for body position
              </li>
              <li className="flex items-center gap-3 text-text-secondary">
                <span className="w-2 h-2 bg-accent-secondary rounded-full"></span>
                21 landmarks per hand (42 total)
              </li>
              <li className="flex items-center gap-3 text-text-secondary">
                <span className="w-2 h-2 bg-accent-tertiary rounded-full"></span>
                Fingertips tracked for precision feedback
              </li>
            </ul>
          </div>
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="rounded-2xl overflow-hidden shadow-lg bg-white p-4"
          >
            <img
              src="/graphics/skeleton-hand.svg"
              alt="Hand skeleton tracking visualization"
              className="w-full"
            />
          </motion.div>
        </motion.div>
      </div>
    </Section>
  );
}

// Applications Section
function ApplicationsSection() {
  const applications = [
    {
      image: "/images/applications/classroom.jpg",
      title: "Classroom Learning",
      desc: "Students practice at their own pace with instant feedback",
    },
    {
      image: "/images/applications/medical.jpg",
      title: "Healthcare Settings",
      desc: "Medical staff communicate with deaf patients effectively",
    },
    {
      image: "/images/applications/responders.png",
      title: "First Responders",
      desc: "Emergency personnel learn critical signs for crisis situations",
    },
    {
      image: "/images/applications/communication.jpg",
      title: "Family Communication",
      desc: "Families connect through shared language learning",
    },
    {
      image: "/images/applications/interpreter.jpg",
      title: "Professional Training",
      desc: "Interpreters get objective assessment for certification",
    },
    {
      image: "/images/applications/research.png",
      title: "Research Applications",
      desc: "Standardized data collection for linguistic studies",
    },
    {
      image: "/images/applications/learning.jpg",
      title: "Self-Paced Study",
      desc: "Learn ASL anywhere with personalized guidance",
    },
    {
      image: "/images/applications/project.png",
      title: "Educational Programs",
      desc: "Schools integrate ASL into curriculum effectively",
    },
  ];

  return (
    <Section className="py-32 bg-bg-secondary" id="applications">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center max-w-3xl mx-auto mb-20">
          <motion.span
            variants={fadeUpVariants}
            className="inline-block text-accent-primary font-semibold tracking-widest text-sm uppercase mb-4"
          >
            Real-World Impact
          </motion.span>
          <motion.h2
            variants={fadeUpVariants}
            className="font-display text-h1 text-text-primary mb-6"
          >
            From Self-Study to{" "}
            <span className="text-gradient">Professional Training</span>
          </motion.h2>
          <motion.p
            variants={fadeUpVariants}
            className="text-text-secondary text-lg"
          >
            SignSense adapts to diverse learning contexts, providing personalized feedback wherever sign language education happens.
          </motion.p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {applications.map((app, i) => (
            <motion.div
              key={i}
              variants={fadeUpVariants}
              className="group relative rounded-2xl overflow-hidden shadow-lg card-hover aspect-[4/3]"
              custom={i}
            >
              {/* Background Image */}
              <img
                src={app.image}
                alt={app.title}
                className="absolute inset-0 w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
              />

              {/* Gradient Overlay */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent" />

              {/* Content */}
              <div className="absolute inset-0 p-6 flex flex-col justify-end">
                <h3 className="font-display text-2xl text-white mb-2 group-hover:text-accent-secondary transition-colors">
                  {app.title}
                </h3>
                <p className="text-white/90 text-sm leading-relaxed opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                  {app.desc}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </Section>
  );
}

// Demo Section
function DemoSection() {
  return (
    <Section className="py-32 bg-bg-primary" id="demo">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <motion.span
            variants={fadeUpVariants}
            className="inline-block text-accent-primary font-semibold tracking-widest text-sm uppercase mb-4"
          >
            See It In Action
          </motion.span>
          <motion.h2
            variants={fadeUpVariants}
            className="font-display text-h1 text-text-primary mb-6"
          >
            Watch SignSense Give{" "}
            <span className="text-gradient">Real-Time Feedback</span>
          </motion.h2>
        </div>

        {/* Video placeholder */}
        <motion.div
          variants={scaleUpVariants}
          className="max-w-4xl mx-auto"
        >
          <div className="relative aspect-video bg-text-primary rounded-3xl overflow-hidden shadow-2xl">
            {/* Placeholder content */}
            <div className="absolute inset-0 flex flex-col items-center justify-center text-white">
              <motion.div
                className="w-24 h-24 rounded-full bg-white/20 flex items-center justify-center mb-6 cursor-pointer"
                whileHover={{ scale: 1.1, backgroundColor: "rgba(255,255,255,0.3)" }}
                whileTap={{ scale: 0.95 }}
              >
                <svg className="w-12 h-12 text-white ml-1" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </motion.div>
              <p className="text-xl font-medium">Demo Video</p>
              <p className="text-white/60 mt-2">Coming Soon</p>
            </div>

            {/* Decorative elements */}
            <div className="absolute bottom-4 left-4 right-4 flex justify-between items-center text-white/60 text-sm">
              <span>0:00</span>
              <div className="flex-1 mx-4 h-1 bg-white/20 rounded-full">
                <div className="w-0 h-full bg-white rounded-full" />
              </div>
              <span>2:30</span>
            </div>
          </div>

          {/* Video chapters */}
          <motion.div
            variants={fadeUpVariants}
            className="mt-8 flex flex-wrap justify-center gap-4"
          >
            {["Introduction", "Practice Mode", "Component Feedback", "Error Correction", "Progress Tracking"].map((chapter, i) => (
              <motion.span
                key={i}
                className="px-4 py-2 bg-bg-secondary rounded-full text-text-secondary text-sm hover:bg-accent-primary/10 hover:text-accent-primary cursor-pointer transition-colors"
                whileHover={{ scale: 1.05 }}
              >
                {chapter}
              </motion.span>
            ))}
          </motion.div>
        </motion.div>
      </div>
    </Section>
  );
}

// Tech Specs Section
function TechSpecsSection() {
  return (
    <Section className="py-32 bg-bg-secondary">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-12">
          {/* Specs table */}
          <motion.div variants={fadeUpVariants}>
            <h3 className="font-display text-h2 text-text-primary mb-8">Technical Specifications</h3>
            <div className="bg-white rounded-2xl overflow-hidden shadow-sm">
              <table className="w-full">
                <thead className="bg-bg-tertiary">
                  <tr>
                    <th className="px-6 py-3 text-left text-sm font-semibold text-text-primary">Model</th>
                    <th className="px-6 py-3 text-right text-sm font-semibold text-text-primary">Parameters</th>
                    <th className="px-6 py-3 text-right text-sm font-semibold text-text-primary">Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { model: "PhonSSM", params: "3.2M", latency: "3.85ms" },
                    { model: "Error Diagnosis", params: "500K", latency: "<1ms" },
                    { model: "Movement Analyzer", params: "100K", latency: "<1ms" },
                    { model: "Feedback Ranker", params: "10K", latency: "<0.1ms" },
                  ].map((row, i) => (
                    <tr key={i} className="border-t border-bg-tertiary">
                      <td className="px-6 py-4 font-medium text-text-primary">{row.model}</td>
                      <td className="px-6 py-4 text-right text-text-secondary font-mono">{row.params}</td>
                      <td className="px-6 py-4 text-right text-text-secondary font-mono">{row.latency}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>

          {/* Privacy callout */}
          <motion.div variants={fadeUpVariants}>
            <h3 className="font-display text-h2 text-text-primary mb-8">Privacy by Design</h3>
            <div className="bg-accent-primary text-white rounded-2xl p-8 space-y-6">
              <div className="flex items-start gap-4">
                <span className="text-3xl">🔒</span>
                <div>
                  <h4 className="font-semibold text-lg">Skeleton-Only Processing</h4>
                  <p className="text-white/80 mt-1">No video is ever stored — only skeleton landmarks</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <span className="text-3xl">👤</span>
                <div>
                  <h4 className="font-semibold text-lg">No Facial Recognition</h4>
                  <p className="text-white/80 mt-1">We don't need or use face data for recognition</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <span className="text-3xl">💻</span>
                <div>
                  <h4 className="font-semibold text-lg">Runs Locally</h4>
                  <p className="text-white/80 mt-1">All processing happens on your device — no cloud required</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <span className="text-3xl">⚡</span>
                <div>
                  <h4 className="font-semibold text-lg">CPU-Only Capable</h4>
                  <p className="text-white/80 mt-1">No GPU needed — works on any modern computer</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </Section>
  );
}

// Footer
function Footer() {
  return (
    <footer className="bg-text-primary text-white py-16">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid md:grid-cols-4 gap-12 mb-12">
          <div className="md:col-span-2">
            <div className="flex items-center gap-3 mb-4">
              <span className="font-display text-3xl text-white">SignSense</span>
            </div>
            <p className="text-white/60 max-w-md leading-relaxed">
              AI-powered sign language learning platform using four specialized neural networks
              for real-time, component-specific feedback.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Research</h4>
            <ul className="space-y-2 text-white/60">
              <li><a href="#" className="hover:text-white transition-colors">Paper (PDF)</a></li>
              <li><a href="#" className="hover:text-white transition-colors">GitHub</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Benchmarks</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Resources</h4>
            <ul className="space-y-2 text-white/60">
              <li><a href="#" className="hover:text-white transition-colors">Documentation</a></li>
              <li><a href="#" className="hover:text-white transition-colors">API Reference</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Tutorials</a></li>
            </ul>
          </div>
        </div>
        <div className="border-t border-white/10 pt-8 flex flex-col md:flex-row justify-between items-center gap-4 text-white/40 text-sm">
          <p>© 2026 SignSense. Built for Toshiba Challenge.</p>
          <div className="flex gap-6">
            <a href="#" className="hover:text-white transition-colors">Privacy Policy</a>
            <a href="#" className="hover:text-white transition-colors">Terms of Service</a>
          </div>
        </div>
      </div>
    </footer>
  );
}

// Main Page
export default function Home() {
  return (
    <main className="relative">
      <Navigation />
      <HeroSection />
      <ProblemSection />
      <SolutionSection />
      <ArchitectureSection />
      <ResultsSection />
      <ApplicationsSection />
      <DemoSection />
      <TechSpecsSection />
      <Footer />
    </main>
  );
}
