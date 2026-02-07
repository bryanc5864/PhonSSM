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

// Skeleton visualization components for hero
function SkeletonJoint({ cx, cy, label, anatomy, delay, isTip }: {
  cx: number; cy: number; label: string; anatomy: string; delay: number; isTip: boolean;
}) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <g style={{ pointerEvents: "all" }}>
      <motion.circle
        cx={cx}
        cy={cy}
        r={isTip ? 6 : 8}
        fill={isTip ? "#C75D4D" : "#E8B86D"}
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.4, delay }}
        style={{ cursor: "pointer", filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.2))" }}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        whileHover={{ scale: 1.4 }}
      />
      {isHovered && (
        <motion.g
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 10 }}
        >
          <rect
            x={cx - 120}
            y={cy - 80}
            width={240}
            height={70}
            rx={12}
            fill="rgba(26, 22, 20, 0.95)"
          />
          <polygon
            points={`${cx - 10},${cy - 10} ${cx + 10},${cy - 10} ${cx},${cy}`}
            fill="rgba(26, 22, 20, 0.95)"
          />
          <text
            x={cx}
            y={cy - 52}
            textAnchor="middle"
            fill="#E8B86D"
            fontSize={18}
            fontWeight={700}
            fontFamily="system-ui, sans-serif"
          >
            {label}
          </text>
          <text
            x={cx}
            y={cy - 28}
            textAnchor="middle"
            fill="white"
            fontSize={14}
            fontFamily="system-ui, sans-serif"
          >
            {anatomy}
          </text>
        </motion.g>
      )}
    </g>
  );
}

function SkeletonBone({ x1, y1, x2, y2, delay }: {
  x1: number; y1: number; x2: number; y2: number; delay: number;
}) {
  return (
    <motion.line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke="#2D5A4A"
      strokeWidth={2}
      strokeLinecap="round"
      initial={{ pathLength: 0, opacity: 0 }}
      animate={{ pathLength: 1, opacity: 0.7 }}
      transition={{ duration: 0.5, delay }}
      style={{ filter: "drop-shadow(0 1px 2px rgba(0,0,0,0.1))" }}
    />
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
  const y = useTransform(scrollYProgress, [0, 1], [0, 100]);
  const opacity = useTransform(scrollYProgress, [0, 0.7, 1], [1, 1, 0]);

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

          {/* Right Visual - Real Hands with Skeleton Overlay */}
          <motion.div
            className="relative"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.4 }}
          >
            <div className="relative w-full aspect-[4/3] max-w-xl mx-auto">
              {/* Glowing background */}
              <div className="absolute inset-0 bg-gradient-radial from-accent-primary/10 via-transparent to-transparent rounded-3xl" />

              {/* Human hands image */}
              <motion.img
                src="/images/human-hand.jpg"
                alt="Both hands with skeleton tracking overlay"
                className="absolute inset-0 w-full h-full object-contain rounded-2xl"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1, delay: 0.5 }}
              />

              {/* Skeleton overlay for BOTH hands - viewBox matches image aspect ratio */}
              <svg
                viewBox="0 0 800 530"
                className="absolute inset-0 w-full h-full"
                style={{ pointerEvents: "none" }}
                preserveAspectRatio="xMidYMid meet"
              >
                {/* ========== LEFT HAND (Palm view) - fingers from left to right: pinky, ring, middle, index, thumb ========== */}
                {/* Left Wrist */}
                <SkeletonJoint cx={200} cy={500} label="L. Wrist" anatomy="Left carpal bones" delay={0.8} isTip={false} />

                {/* Left Pinky (leftmost finger) */}
                <SkeletonBone x1={200} y1={500} x2={125} y2={400} delay={0.9} />
                <SkeletonBone x1={125} y1={400} x2={100} y2={290} delay={1.0} />
                <SkeletonBone x1={100} y1={290} x2={90} y2={190} delay={1.1} />
                <SkeletonBone x1={90} y1={190} x2={85} y2={105} delay={1.2} />
                <SkeletonJoint cx={125} cy={400} label="MCP" anatomy="Pinky knuckle" delay={1.0} isTip={false} />
                <SkeletonJoint cx={100} cy={290} label="PIP" anatomy="First joint" delay={1.1} isTip={false} />
                <SkeletonJoint cx={90} cy={190} label="DIP" anatomy="Second joint" delay={1.2} isTip={false} />
                <SkeletonJoint cx={85} cy={105} label="Tip" anatomy="Pinky tip" delay={1.3} isTip={true} />

                {/* Left Ring */}
                <SkeletonBone x1={200} y1={500} x2={158} y2={385} delay={1.0} />
                <SkeletonBone x1={158} y1={385} x2={142} y2={265} delay={1.1} />
                <SkeletonBone x1={142} y1={265} x2={135} y2={160} delay={1.2} />
                <SkeletonBone x1={135} y1={160} x2={130} y2={70} delay={1.3} />
                <SkeletonJoint cx={158} cy={385} label="MCP" anatomy="Ring knuckle" delay={1.1} isTip={false} />
                <SkeletonJoint cx={142} cy={265} label="PIP" anatomy="First joint" delay={1.2} isTip={false} />
                <SkeletonJoint cx={135} cy={160} label="DIP" anatomy="Second joint" delay={1.3} isTip={false} />
                <SkeletonJoint cx={130} cy={70} label="Tip" anatomy="Ring tip" delay={1.4} isTip={true} />

                {/* Left Middle */}
                <SkeletonBone x1={200} y1={500} x2={192} y2={380} delay={1.1} />
                <SkeletonBone x1={192} y1={380} x2={185} y2={255} delay={1.2} />
                <SkeletonBone x1={185} y1={255} x2={180} y2={145} delay={1.3} />
                <SkeletonBone x1={180} y1={145} x2={178} y2={50} delay={1.4} />
                <SkeletonJoint cx={192} cy={380} label="MCP" anatomy="Middle knuckle" delay={1.2} isTip={false} />
                <SkeletonJoint cx={185} cy={255} label="PIP" anatomy="First joint" delay={1.3} isTip={false} />
                <SkeletonJoint cx={180} cy={145} label="DIP" anatomy="Second joint" delay={1.4} isTip={false} />
                <SkeletonJoint cx={178} cy={50} label="Tip" anatomy="Middle tip" delay={1.5} isTip={true} />

                {/* Left Index */}
                <SkeletonBone x1={200} y1={500} x2={228} y2={385} delay={1.2} />
                <SkeletonBone x1={228} y1={385} x2={238} y2={270} delay={1.3} />
                <SkeletonBone x1={238} y1={270} x2={245} y2={170} delay={1.4} />
                <SkeletonBone x1={245} y1={170} x2={250} y2={85} delay={1.5} />
                <SkeletonJoint cx={228} cy={385} label="MCP" anatomy="Index knuckle" delay={1.3} isTip={false} />
                <SkeletonJoint cx={238} cy={270} label="PIP" anatomy="First joint" delay={1.4} isTip={false} />
                <SkeletonJoint cx={245} cy={170} label="DIP" anatomy="Second joint" delay={1.5} isTip={false} />
                <SkeletonJoint cx={250} cy={85} label="Tip" anatomy="Index tip" delay={1.6} isTip={true} />

                {/* Left Thumb (on right side of left palm) */}
                <SkeletonBone x1={200} y1={500} x2={265} y2={450} delay={1.3} />
                <SkeletonBone x1={265} y1={450} x2={295} y2={400} delay={1.4} />
                <SkeletonBone x1={295} y1={400} x2={315} y2={360} delay={1.5} />
                <SkeletonJoint cx={265} cy={450} label="CMC" anatomy="Thumb base" delay={1.4} isTip={false} />
                <SkeletonJoint cx={295} cy={400} label="MCP" anatomy="Thumb knuckle" delay={1.5} isTip={false} />
                <SkeletonJoint cx={315} cy={360} label="Tip" anatomy="Thumb tip" delay={1.6} isTip={true} />

                {/* ========== RIGHT HAND (Back view) - fingers from left to right: thumb, index, middle, ring, pinky ========== */}
                {/* Right Wrist */}
                <SkeletonJoint cx={600} cy={505} label="R. Wrist" anatomy="Right carpal bones" delay={1.0} isTip={false} />

                {/* Right Thumb (on left side of right hand) */}
                <SkeletonBone x1={600} y1={505} x2={530} y2={455} delay={1.1} />
                <SkeletonBone x1={530} y1={455} x2={490} y2={405} delay={1.2} />
                <SkeletonBone x1={490} y1={405} x2={460} y2={365} delay={1.3} />
                <SkeletonJoint cx={530} cy={455} label="CMC" anatomy="Thumb base" delay={1.2} isTip={false} />
                <SkeletonJoint cx={490} cy={405} label="MCP" anatomy="Thumb knuckle" delay={1.3} isTip={false} />
                <SkeletonJoint cx={460} cy={365} label="Tip" anatomy="Thumb tip" delay={1.4} isTip={true} />

                {/* Right Index */}
                <SkeletonBone x1={600} y1={505} x2={548} y2={385} delay={1.2} />
                <SkeletonBone x1={548} y1={385} x2={520} y2={270} delay={1.3} />
                <SkeletonBone x1={520} y1={270} x2={505} y2={170} delay={1.4} />
                <SkeletonBone x1={505} y1={170} x2={495} y2={90} delay={1.5} />
                <SkeletonJoint cx={548} cy={385} label="MCP" anatomy="Index knuckle" delay={1.3} isTip={false} />
                <SkeletonJoint cx={520} cy={270} label="PIP" anatomy="First joint" delay={1.4} isTip={false} />
                <SkeletonJoint cx={505} cy={170} label="DIP" anatomy="Second joint" delay={1.5} isTip={false} />
                <SkeletonJoint cx={495} cy={90} label="Tip" anatomy="Index tip" delay={1.6} isTip={true} />

                {/* Right Middle */}
                <SkeletonBone x1={600} y1={505} x2={595} y2={380} delay={1.3} />
                <SkeletonBone x1={595} y1={380} x2={590} y2={260} delay={1.4} />
                <SkeletonBone x1={590} y1={260} x2={587} y2={155} delay={1.5} />
                <SkeletonBone x1={587} y1={155} x2={585} y2={65} delay={1.6} />
                <SkeletonJoint cx={595} cy={380} label="MCP" anatomy="Middle knuckle" delay={1.4} isTip={false} />
                <SkeletonJoint cx={590} cy={260} label="PIP" anatomy="First joint" delay={1.5} isTip={false} />
                <SkeletonJoint cx={587} cy={155} label="DIP" anatomy="Second joint" delay={1.6} isTip={false} />
                <SkeletonJoint cx={585} cy={65} label="Tip" anatomy="Middle tip" delay={1.7} isTip={true} />

                {/* Right Ring */}
                <SkeletonBone x1={600} y1={505} x2={640} y2={385} delay={1.4} />
                <SkeletonBone x1={640} y1={385} x2={660} y2={275} delay={1.5} />
                <SkeletonBone x1={660} y1={275} x2={675} y2={175} delay={1.6} />
                <SkeletonBone x1={675} y1={175} x2={685} y2={95} delay={1.7} />
                <SkeletonJoint cx={640} cy={385} label="MCP" anatomy="Ring knuckle" delay={1.5} isTip={false} />
                <SkeletonJoint cx={660} cy={275} label="PIP" anatomy="First joint" delay={1.6} isTip={false} />
                <SkeletonJoint cx={675} cy={175} label="DIP" anatomy="Second joint" delay={1.7} isTip={false} />
                <SkeletonJoint cx={685} cy={95} label="Tip" anatomy="Ring tip" delay={1.8} isTip={true} />

                {/* Right Pinky (rightmost finger) */}
                <SkeletonBone x1={600} y1={505} x2={675} y2={400} delay={1.5} />
                <SkeletonBone x1={675} y1={400} x2={705} y2={310} delay={1.6} />
                <SkeletonBone x1={705} y1={310} x2={725} y2={230} delay={1.7} />
                <SkeletonBone x1={725} y1={230} x2={740} y2={160} delay={1.8} />
                <SkeletonJoint cx={675} cy={400} label="MCP" anatomy="Pinky knuckle" delay={1.6} isTip={false} />
                <SkeletonJoint cx={705} cy={310} label="PIP" anatomy="First joint" delay={1.7} isTip={false} />
                <SkeletonJoint cx={725} cy={230} label="DIP" anatomy="Second joint" delay={1.8} isTip={false} />
                <SkeletonJoint cx={740} cy={160} label="Tip" anatomy="Pinky tip" delay={1.9} isTip={true} />
              </svg>

              {/* Floating labels */}
              {[
                { label: "Palm View", x: "8%", y: "8%", delay: 2 },
                { label: "Back View", x: "60%", y: "8%", delay: 2.2 },
                { label: "42 hand landmarks", x: "35%", y: "92%", delay: 2.4 },
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
                { num: "1", color: "bg-accent-tertiary", title: "Limited Access", desc: "Qualified ASL instructors are scarce, especially outside urban areas" },
                { num: "2", color: "bg-accent-secondary", title: "No Real-Time Feedback", desc: "Books and videos can't tell you if you're signing correctly" },
                { num: "3", color: "bg-text-tertiary", title: "Binary Assessment", desc: "Traditional apps only say 'right' or 'wrong' — not what to fix" },
              ].map((item, i) => (
                <motion.div
                  key={i}
                  className="flex gap-4 p-5 bg-white rounded-xl shadow-sm border border-bg-tertiary"
                  variants={fadeUpVariants}
                  whileHover={{ x: 10, boxShadow: "0 10px 30px rgba(0,0,0,0.08)", borderColor: "rgba(45, 90, 74, 0.2)" }}
                  transition={{ duration: 0.2 }}
                >
                  <div className={`w-12 h-12 ${item.color} rounded-lg flex items-center justify-center flex-shrink-0`}>
                    <span className="text-white font-bold text-lg">{item.num}</span>
                  </div>
                  <div>
                    <h4 className="font-semibold text-text-primary text-lg">{item.title}</h4>
                    <p className="text-text-secondary mt-1">{item.desc}</p>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>

          {/* Comparison Visual */}
          <motion.div variants={scaleUpVariants} className="relative">
            {/* ASL Learning Image */}
            <motion.div
              className="mb-6 rounded-2xl overflow-hidden shadow-lg"
              whileHover={{ scale: 1.02 }}
              transition={{ duration: 0.3 }}
            >
              <img
                src="/images/asl-learning.jpg"
                alt="People learning sign language"
                className="w-full h-64 object-cover"
              />
            </motion.div>

            <div className="bg-white rounded-2xl shadow-xl p-8 space-y-6">
              <h3 className="font-display text-2xl text-center text-text-primary mb-8">
                The SignSense Difference
              </h3>

              <div className="grid grid-cols-2 gap-6">
                {/* Traditional */}
                <div className="space-y-4">
                  <div className="text-center p-4 bg-red-50 rounded-xl">
                    <div className="w-12 h-12 mx-auto bg-accent-tertiary/20 rounded-lg flex items-center justify-center">
                      <svg className="w-6 h-6 text-accent-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <h4 className="font-semibold mt-3 text-text-primary">Traditional Apps</h4>
                  </div>
                  <div className="p-4 bg-red-100/50 rounded-lg text-center">
                    <div className="w-10 h-10 mx-auto bg-error/20 rounded-full flex items-center justify-center">
                      <svg className="w-5 h-5 text-error" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </div>
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
              <div className="w-10 h-10 mx-auto bg-accent-primary/20 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-accent-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
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
              <div className="w-10 h-10 mx-auto bg-accent-primary rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
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

  // Animated bar data for background - more bars for visual richness
  const bars = [
    { width: "92%", height: "h-16", top: "5%", delay: 0, color: "bg-accent-primary" },
    { width: "65%", height: "h-20", top: "18%", delay: 0.15, color: "bg-accent-secondary" },
    { width: "78%", height: "h-14", top: "32%", delay: 0.3, color: "bg-accent-primary" },
    { width: "55%", height: "h-24", top: "48%", delay: 0.45, color: "bg-accent-tertiary" },
    { width: "85%", height: "h-12", top: "62%", delay: 0.6, color: "bg-accent-primary" },
    { width: "70%", height: "h-18", top: "76%", delay: 0.75, color: "bg-accent-secondary" },
    { width: "45%", height: "h-16", top: "88%", delay: 0.9, color: "bg-accent-primary" },
  ];

  return (
    <Section className="py-32 bg-bg-primary relative overflow-hidden" id="results">
      {/* Animated Background Bars - drifting chart effect */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {bars.map((bar, i) => (
          <motion.div
            key={i}
            className={`absolute ${bar.height} ${bar.color} rounded-r-full`}
            style={{ top: bar.top, opacity: 0.06 }}
            initial={{ width: "0%", x: "-50%" }}
            whileInView={{ width: bar.width, x: "0%" }}
            transition={{
              duration: 2.5,
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

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {applications.map((app, i) => (
            <motion.div
              key={i}
              variants={fadeUpVariants}
              className="group relative rounded-2xl overflow-hidden shadow-lg card-hover aspect-[3/2]"
              custom={i}
              whileHover={{ y: -8 }}
              transition={{ duration: 0.3 }}
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

        {/* Video placeholder with ASL preview */}
        <motion.div
          variants={scaleUpVariants}
          className="max-w-4xl mx-auto"
        >
          <div className="relative aspect-video rounded-3xl overflow-hidden shadow-2xl">
            {/* Background image */}
            <img
              src="/images/asl-examples.png"
              alt="ASL sign examples grid"
              className="absolute inset-0 w-full h-full object-cover"
            />
            {/* Dark overlay */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-black/20" />

            {/* Play button content */}
            <div className="absolute inset-0 flex flex-col items-center justify-center text-white">
              <motion.div
                className="w-24 h-24 rounded-full bg-accent-primary/90 flex items-center justify-center mb-6 cursor-pointer shadow-xl"
                whileHover={{ scale: 1.1, boxShadow: "0 25px 50px rgba(45, 90, 74, 0.4)" }}
                whileTap={{ scale: 0.95 }}
                animate={{ scale: [1, 1.05, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <svg className="w-12 h-12 text-white ml-1" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </motion.div>
              <p className="text-2xl font-display font-medium">Watch Demo</p>
              <p className="text-white/70 mt-2">See SignSense in action</p>
            </div>

            {/* Decorative elements */}
            <div className="absolute bottom-4 left-4 right-4 flex justify-between items-center text-white/60 text-sm">
              <span>0:00</span>
              <div className="flex-1 mx-4 h-1 bg-white/20 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-accent-primary rounded-full"
                  initial={{ width: "0%" }}
                  whileInView={{ width: "30%" }}
                  transition={{ duration: 2, delay: 0.5 }}
                  viewport={{ once: true }}
                />
              </div>
              <span>2:30</span>
            </div>
          </div>

          {/* Video chapters with better styling */}
          <motion.div
            variants={fadeUpVariants}
            className="mt-8 flex flex-wrap justify-center gap-3"
          >
            {[
              { name: "Introduction", time: "0:00" },
              { name: "Practice Mode", time: "0:30" },
              { name: "Component Feedback", time: "1:00" },
              { name: "Error Correction", time: "1:45" },
              { name: "Progress Tracking", time: "2:15" },
            ].map((chapter, i) => (
              <motion.div
                key={i}
                className="px-4 py-2 bg-white rounded-full shadow-sm border border-bg-tertiary text-text-secondary text-sm hover:bg-accent-primary hover:text-white hover:border-accent-primary cursor-pointer transition-all flex items-center gap-2"
                whileHover={{ scale: 1.05, y: -2 }}
              >
                <span className="text-xs text-text-tertiary">{chapter.time}</span>
                <span>{chapter.name}</span>
              </motion.div>
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
            <div className="bg-accent-primary text-white rounded-2xl p-8 space-y-5">
              {[
                {
                  icon: <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />,
                  title: "Skeleton-Only Processing",
                  desc: "No video is ever stored — only skeleton landmarks"
                },
                {
                  icon: <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />,
                  title: "No Facial Recognition",
                  desc: "We don't need or use face data for recognition"
                },
                {
                  icon: <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />,
                  title: "Runs Locally",
                  desc: "All processing happens on your device — no cloud required"
                },
                {
                  icon: <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />,
                  title: "CPU-Only Capable",
                  desc: "No GPU needed — works on any modern computer"
                },
              ].map((item, i) => (
                <motion.div
                  key={i}
                  className="flex items-start gap-4"
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  viewport={{ once: true }}
                >
                  <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      {item.icon}
                    </svg>
                  </div>
                  <div>
                    <h4 className="font-semibold text-lg">{item.title}</h4>
                    <p className="text-white/80 mt-1">{item.desc}</p>
                  </div>
                </motion.div>
              ))}
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
