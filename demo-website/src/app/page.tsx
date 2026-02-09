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

// Video tile showing sign with prediction label (optimized for performance)
function VideoTile({ videoSrc, prediction, confidence, index }: {
  videoSrc: string; prediction: string; confidence: number; index: number;
}) {
  return (
    <div className="video-tile aspect-[3/4] bg-bg-tertiary relative overflow-hidden rounded-lg">
      <video
        src={videoSrc}
        autoPlay
        loop
        muted
        playsInline
        className="w-full h-full object-cover"
      />
      {/* Simple gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black/40 pointer-events-none" />
      {/* Prediction badge */}
      <div className="absolute bottom-1 left-1 right-1">
        <div className="bg-black/70 rounded px-1.5 py-0.5">
          <div className="flex items-center justify-between gap-1">
            <span className="text-white text-[9px] font-bold uppercase tracking-wide truncate">{prediction}</span>
            <span className="text-accent-primary text-[9px] font-semibold">{confidence}%</span>
          </div>
        </div>
      </div>
    </div>
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
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 sm:py-4 flex justify-between items-center">
        <motion.div
          className="flex items-center gap-2"
          whileHover={{ scale: 1.02 }}
        >
          <img src="/logo.svg" alt="SignSense" className="h-8 sm:h-10" />
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

// Hero Section with Video Background Grid - 58 diverse signers from ASL Citizen
function HeroSection() {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const y = useTransform(scrollYProgress, [0, 1], [0, 100]);
  const opacity = useTransform(scrollYProgress, [0, 0.7, 1], [1, 1, 0]);

  // 58 diverse sign videos from ASL Citizen dataset - different signers, common vocabulary
  const videoTiles = [
    { src: "/videos/signs/hello.mp4", prediction: "HELLO", confidence: 96 },
    { src: "/videos/signs/thank_you.mp4", prediction: "THANK YOU", confidence: 94 },
    { src: "/videos/signs/please.mp4", prediction: "PLEASE", confidence: 91 },
    { src: "/videos/signs/help.mp4", prediction: "HELP", confidence: 89 },
    { src: "/videos/signs/yes.mp4", prediction: "YES", confidence: 97 },
    { src: "/videos/signs/no.mp4", prediction: "NO", confidence: 95 },
    { src: "/videos/signs/good.mp4", prediction: "GOOD", confidence: 93 },
    { src: "/videos/signs/bad.mp4", prediction: "BAD", confidence: 88 },
    { src: "/videos/signs/friend.mp4", prediction: "FRIEND", confidence: 92 },
    { src: "/videos/signs/family.mp4", prediction: "FAMILY", confidence: 90 },
    { src: "/videos/signs/mother.mp4", prediction: "MOTHER", confidence: 94 },
    { src: "/videos/signs/father.mp4", prediction: "FATHER", confidence: 93 },
    { src: "/videos/signs/love.mp4", prediction: "LOVE", confidence: 96 },
    { src: "/videos/signs/happy.mp4", prediction: "HAPPY", confidence: 91 },
    { src: "/videos/signs/sad.mp4", prediction: "SAD", confidence: 89 },
    { src: "/videos/signs/sorry.mp4", prediction: "SORRY", confidence: 87 },
    { src: "/videos/signs/work.mp4", prediction: "WORK", confidence: 92 },
    { src: "/videos/signs/school.mp4", prediction: "SCHOOL", confidence: 90 },
    { src: "/videos/signs/eat.mp4", prediction: "EAT", confidence: 95 },
    { src: "/videos/signs/drink.mp4", prediction: "DRINK", confidence: 93 },
    { src: "/videos/signs/water.mp4", prediction: "WATER", confidence: 94 },
    { src: "/videos/signs/home.mp4", prediction: "HOME", confidence: 91 },
    { src: "/videos/signs/want.mp4", prediction: "WANT", confidence: 88 },
    { src: "/videos/signs/need.mp4", prediction: "NEED", confidence: 86 },
    { src: "/videos/signs/like.mp4", prediction: "LIKE", confidence: 90 },
    { src: "/videos/signs/name.mp4", prediction: "NAME", confidence: 92 },
    { src: "/videos/signs/where.mp4", prediction: "WHERE", confidence: 89 },
    { src: "/videos/signs/who.mp4", prediction: "WHO", confidence: 91 },
    { src: "/videos/signs/when.mp4", prediction: "WHEN", confidence: 87 },
    { src: "/videos/signs/why.mp4", prediction: "WHY", confidence: 88 },
    { src: "/videos/signs/how.mp4", prediction: "HOW", confidence: 90 },
    { src: "/videos/signs/sleep.mp4", prediction: "SLEEP", confidence: 94 },
    { src: "/videos/signs/play.mp4", prediction: "PLAY", confidence: 91 },
    { src: "/videos/signs/learn.mp4", prediction: "LEARN", confidence: 89 },
    { src: "/videos/signs/think.mp4", prediction: "THINK", confidence: 86 },
    { src: "/videos/signs/know.mp4", prediction: "KNOW", confidence: 92 },
    { src: "/videos/signs/understand.mp4", prediction: "UNDERSTAND", confidence: 85 },
    { src: "/videos/signs/remember.mp4", prediction: "REMEMBER", confidence: 84 },
    { src: "/videos/signs/see.mp4", prediction: "SEE", confidence: 93 },
    { src: "/videos/signs/feel.mp4", prediction: "FEEL", confidence: 87 },
    { src: "/videos/signs/give.mp4", prediction: "GIVE", confidence: 91 },
    { src: "/videos/signs/make.mp4", prediction: "MAKE", confidence: 88 },
    { src: "/videos/signs/open.mp4", prediction: "OPEN", confidence: 90 },
    { src: "/videos/signs/close.mp4", prediction: "CLOSE", confidence: 89 },
    { src: "/videos/signs/start.mp4", prediction: "START", confidence: 87 },
    { src: "/videos/signs/stop.mp4", prediction: "STOP", confidence: 94 },
    { src: "/videos/signs/wait.mp4", prediction: "WAIT", confidence: 92 },
    { src: "/videos/signs/finish.mp4", prediction: "FINISH", confidence: 90 },
    { src: "/videos/signs/try.mp4", prediction: "TRY", confidence: 86 },
    { src: "/videos/signs/find.mp4", prediction: "FIND", confidence: 88 },
    { src: "/videos/signs/tell.mp4", prediction: "TELL", confidence: 89 },
    { src: "/videos/signs/meet.mp4", prediction: "MEET", confidence: 91 },
    { src: "/videos/signs/walk.mp4", prediction: "WALK", confidence: 93 },
    { src: "/videos/signs/sit.mp4", prediction: "SIT", confidence: 95 },
    { src: "/videos/signs/stand.mp4", prediction: "STAND", confidence: 92 },
    { src: "/videos/signs/big.mp4", prediction: "BIG", confidence: 96 },
    { src: "/videos/signs/small.mp4", prediction: "SMALL", confidence: 94 },
    { src: "/videos/signs/funny.mp4", prediction: "FUNNY", confidence: 88 },
  ];

  return (
    <section ref={ref} className="min-h-screen relative overflow-hidden">
      {/* Video Grid Background - 58 diverse signers, GPU accelerated */}
      <div className="absolute inset-0 z-0">
        {/* Dark gradient overlay for readability */}
        <div className="absolute inset-0 bg-gradient-to-b from-bg-primary/90 via-bg-primary/70 to-bg-primary/90 z-10" />

        {/* Video grid - responsive: 4 cols mobile, 6 tablet, 10 desktop */}
        <div className="absolute inset-0 grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-10 gap-0.5 p-0.5 opacity-60">
          {videoTiles.map((tile, i) => (
            <VideoTile
              key={i}
              videoSrc={tile.src}
              prediction={tile.prediction}
              confidence={tile.confidence}
              index={i}
            />
          ))}
        </div>
      </div>

      {/* Hero Content */}
      <motion.div
        className="relative z-20 max-w-7xl mx-auto px-4 sm:px-6 pt-24 sm:pt-32 pb-16 sm:pb-20 min-h-screen flex flex-col justify-center"
        style={{ y, opacity }}
      >
        <div className="max-w-3xl mx-auto text-center space-y-4 sm:space-y-6 lg:space-y-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <span className="inline-block px-3 py-1.5 sm:px-4 sm:py-2 bg-accent-primary/10 backdrop-blur-sm text-accent-primary rounded-full text-xs sm:text-sm font-medium mb-4 sm:mb-6">
              Toshiba Challenge 2026
            </span>
          </motion.div>

          <motion.h1
            className="font-display text-3xl sm:text-4xl md:text-5xl lg:text-hero text-text-primary leading-tight px-2"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
          >
            Learn Sign Language with{" "}
            <span className="text-gradient">AI That Actually Understands</span>
          </motion.h1>

          <motion.p
            className="text-base sm:text-lg lg:text-xl text-text-secondary max-w-2xl mx-auto leading-relaxed px-2"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            SignSense uses <strong>four specialized neural networks</strong> working together
            to give you real-time, component-specific feedback on your signing technique.
            Not just "right" or "wrong" — but exactly what to fix.
          </motion.p>

          <motion.div
            className="flex flex-col sm:flex-row flex-wrap gap-3 sm:gap-4 pt-2 sm:pt-4 justify-center px-4"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            <motion.a
              href="#demo"
              className="inline-flex items-center justify-center gap-2 bg-accent-gradient text-white px-6 sm:px-8 py-3 sm:py-4 rounded-full font-semibold text-base sm:text-lg shadow-lg"
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
              className="inline-flex items-center justify-center gap-2 border-2 border-accent-primary text-accent-primary px-6 sm:px-8 py-3 sm:py-4 rounded-full font-semibold text-base sm:text-lg bg-white/80 backdrop-blur-sm"
              whileHover={{ backgroundColor: "rgba(45, 90, 74, 0.1)" }}
              whileTap={{ scale: 0.98 }}
            >
              Explore Technology
            </motion.a>
          </motion.div>

          {/* Quick Stats */}
          <motion.div
            className="flex flex-wrap gap-3 sm:gap-4 lg:gap-8 pt-4 sm:pt-8 justify-center px-2"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.5 }}
          >
            {[
              { value: "88.4", label: "% Accuracy" },
              { value: "4", label: "AI Models" },
              { value: "5,565", label: "Signs" },
            ].map((stat, i) => (
              <div key={i} className="text-center bg-white/80 backdrop-blur-sm rounded-lg sm:rounded-xl px-3 sm:px-5 lg:px-6 py-2 sm:py-3 lg:py-4 shadow-sm">
                <div className="text-xl sm:text-2xl lg:text-3xl font-bold text-accent-primary">{stat.value}</div>
                <div className="text-xs sm:text-sm text-text-tertiary">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </div>

        {/* Scroll indicator - hidden on mobile */}
        <motion.div
          className="absolute bottom-4 sm:bottom-8 left-1/2 -translate-x-1/2 hidden sm:block"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1, y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity, delay: 2 }}
        >
          <div className="flex flex-col items-center text-text-tertiary bg-white/80 backdrop-blur-sm rounded-full px-3 py-1.5 sm:px-4 sm:py-2">
            <span className="text-xs sm:text-sm mb-1 sm:mb-2">Scroll to explore</span>
            <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
    <Section className="py-16 sm:py-24 lg:py-32 bg-bg-secondary" id="problem">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="grid lg:grid-cols-2 gap-8 sm:gap-12 lg:gap-16 items-center">
          <div className="space-y-4 sm:space-y-6 lg:space-y-8">
            <motion.span
              variants={fadeUpVariants}
              className="inline-block text-accent-primary font-semibold tracking-widest text-xs sm:text-sm uppercase"
            >
              The Challenge
            </motion.span>

            <motion.h2
              variants={fadeUpVariants}
              className="font-display text-2xl sm:text-3xl lg:text-h1 text-text-primary"
            >
              70 Million People Use Sign Language.{" "}
              <span className="text-accent-tertiary">Learning It Shouldn't Be This Hard.</span>
            </motion.h2>

            <motion.div variants={fadeUpVariants} className="space-y-3 sm:space-y-4 lg:space-y-6">
              {[
                { num: "1", color: "bg-accent-tertiary", title: "Limited Access", desc: "Qualified ASL instructors are scarce, especially outside urban areas" },
                { num: "2", color: "bg-accent-secondary", title: "No Real-Time Feedback", desc: "Books and videos can't tell you if you're signing correctly" },
                { num: "3", color: "bg-text-tertiary", title: "Binary Assessment", desc: "Traditional apps only say 'right' or 'wrong' — not what to fix" },
              ].map((item, i) => (
                <motion.div
                  key={i}
                  className="flex gap-3 sm:gap-4 p-3 sm:p-4 lg:p-5 bg-white rounded-lg sm:rounded-xl shadow-sm border border-bg-tertiary"
                  variants={fadeUpVariants}
                  whileHover={{ x: 10, boxShadow: "0 10px 30px rgba(0,0,0,0.08)", borderColor: "rgba(45, 90, 74, 0.2)" }}
                  transition={{ duration: 0.2 }}
                >
                  <div className={`w-10 h-10 sm:w-12 sm:h-12 ${item.color} rounded-lg flex items-center justify-center flex-shrink-0`}>
                    <span className="text-white font-bold text-base sm:text-lg">{item.num}</span>
                  </div>
                  <div>
                    <h4 className="font-semibold text-text-primary text-base sm:text-lg">{item.title}</h4>
                    <p className="text-text-secondary text-sm sm:text-base mt-0.5 sm:mt-1">{item.desc}</p>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>

          {/* Comparison Visual */}
          <motion.div variants={scaleUpVariants} className="relative">
            {/* ASL Learning Image */}
            <motion.div
              className="mb-4 sm:mb-6 rounded-xl sm:rounded-2xl overflow-hidden shadow-lg"
              whileHover={{ scale: 1.02 }}
              transition={{ duration: 0.3 }}
            >
              <img
                src="/images/asl-learning.jpg"
                alt="People learning sign language"
                className="w-full h-48 sm:h-56 lg:h-64 object-cover"
              />
            </motion.div>

            <div className="bg-white rounded-xl sm:rounded-2xl shadow-xl p-4 sm:p-6 lg:p-8 space-y-4 sm:space-y-6">
              <h3 className="font-display text-lg sm:text-xl lg:text-2xl text-center text-text-primary mb-4 sm:mb-6 lg:mb-8">
                The SignSense Difference
              </h3>

              <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:gap-6">
                {/* Traditional */}
                <div className="space-y-3 sm:space-y-4">
                  <div className="text-center p-3 sm:p-4 bg-red-50 rounded-lg sm:rounded-xl">
                    <div className="w-10 h-10 sm:w-12 sm:h-12 mx-auto bg-accent-tertiary/20 rounded-lg flex items-center justify-center">
                      <svg className="w-5 h-5 sm:w-6 sm:h-6 text-accent-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <h4 className="font-semibold mt-2 sm:mt-3 text-text-primary text-sm sm:text-base">Traditional Apps</h4>
                  </div>
                  <div className="p-3 sm:p-4 bg-red-100/50 rounded-lg text-center">
                    <div className="w-8 h-8 sm:w-10 sm:h-10 mx-auto bg-error/20 rounded-full flex items-center justify-center">
                      <svg className="w-4 h-4 sm:w-5 sm:h-5 text-error" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </div>
                    <p className="text-error font-semibold mt-1.5 sm:mt-2 text-sm sm:text-base">Wrong</p>
                    <p className="text-xs sm:text-sm text-text-secondary mt-0.5 sm:mt-1">No explanation why</p>
                  </div>
                </div>

                {/* SignSense */}
                <div className="space-y-3 sm:space-y-4">
                  <div className="text-center p-3 sm:p-4 bg-accent-primary/10 rounded-lg sm:rounded-xl">
                    <span className="text-lg sm:text-xl lg:text-2xl font-bold text-accent-primary">SignSense</span>
                  </div>
                  <div className="p-3 sm:p-4 bg-accent-primary/5 rounded-lg">
                    <p className="text-success font-bold text-sm sm:text-base lg:text-lg">Correct!</p>
                    <div className="mt-2 sm:mt-3 space-y-1.5 sm:space-y-2 text-xs sm:text-sm">
                      {[
                        { label: "Handshape", value: 94 },
                        { label: "Location", value: 87 },
                        { label: "Movement", value: 91 },
                      ].map((item, i) => (
                        <div key={i} className="flex items-center gap-1 sm:gap-2">
                          <span className="text-text-secondary w-14 sm:w-20 text-[10px] sm:text-xs">{item.label}</span>
                          <div className="flex-1 bg-bg-tertiary rounded-full h-1.5 sm:h-2">
                            <motion.div
                              className="bg-accent-primary h-full rounded-full"
                              initial={{ width: 0 }}
                              whileInView={{ width: `${item.value}%` }}
                              transition={{ duration: 1, delay: i * 0.2 }}
                            />
                          </div>
                          <span className="text-accent-primary font-medium w-7 sm:w-10 text-[10px] sm:text-xs">{item.value}%</span>
                        </div>
                      ))}
                    </div>
                    <p className="mt-2 sm:mt-3 text-[10px] sm:text-xs text-accent-primary bg-accent-primary/10 p-1.5 sm:p-2 rounded font-medium">
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
    <Section className="py-16 sm:py-24 lg:py-32 bg-bg-primary" id="technology">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="text-center max-w-3xl mx-auto mb-10 sm:mb-16 lg:mb-20 px-2">
          <motion.span
            variants={fadeUpVariants}
            className="inline-block text-accent-primary font-semibold tracking-widest text-xs sm:text-sm uppercase mb-2 sm:mb-4"
          >
            Our Approach
          </motion.span>
          <motion.h2
            variants={fadeUpVariants}
            className="font-display text-2xl sm:text-3xl lg:text-h1 text-text-primary mb-4 sm:mb-6"
          >
            Four Specialized Models.{" "}
            <span className="text-gradient">One Seamless Experience.</span>
          </motion.h2>
          <motion.p
            variants={fadeUpVariants}
            className="text-base sm:text-lg lg:text-xl text-text-secondary"
          >
            Unlike simple classifiers, SignSense employs a diagnostic pipeline that identifies
            exactly what you need to fix — not just that something is wrong.
          </motion.p>
        </div>

        {/* Model cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 mb-10 sm:mb-16">
          {models.map((model, i) => (
            <motion.div
              key={i}
              variants={fadeUpVariants}
              className="bg-white rounded-xl sm:rounded-2xl p-4 sm:p-6 shadow-sm card-hover border border-transparent hover:border-accent-primary/20"
              custom={i}
            >
              <motion.div
                className={`w-10 h-10 sm:w-12 sm:h-12 lg:w-14 lg:h-14 ${model.bgColor} rounded-lg sm:rounded-xl flex items-center justify-center mb-3 sm:mb-4`}
                whileHover={{ scale: 1.1, rotate: 5 }}
                transition={{ type: "spring", stiffness: 400 }}
              >
                <span className="text-white text-lg sm:text-xl lg:text-2xl font-bold">{model.num}</span>
              </motion.div>
              <h3 className="text-xs sm:text-sm font-bold tracking-wider text-accent-primary mb-0.5 sm:mb-1">
                {model.title}
              </h3>
              <h4 className="font-display text-base sm:text-lg lg:text-xl text-text-primary mb-2 sm:mb-3">
                {model.subtitle}
              </h4>
              <p className="text-text-secondary text-xs sm:text-sm leading-relaxed">
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
          className="mt-10 sm:mt-16 lg:mt-20"
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
    <Section className="py-16 sm:py-24 lg:py-32 bg-bg-secondary" id="architecture">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="text-center max-w-3xl mx-auto mb-10 sm:mb-16 lg:mb-20 px-2">
          <motion.span
            variants={fadeUpVariants}
            className="inline-block text-accent-primary font-semibold tracking-widest text-xs sm:text-sm uppercase mb-2 sm:mb-4"
          >
            Core Technology
          </motion.span>
          <motion.h2
            variants={fadeUpVariants}
            className="font-display text-2xl sm:text-3xl lg:text-h1 text-text-primary mb-4 sm:mb-6"
          >
            Built on 60 Years of{" "}
            <span className="text-gradient">Sign Language Linguistics</span>
          </motion.h2>
          <motion.p
            variants={fadeUpVariants}
            className="text-base sm:text-lg lg:text-xl text-text-secondary"
          >
            PhonSSM's architecture embeds Stokoe's phonological theory directly into the neural network,
            enabling unprecedented accuracy and interpretable feedback.
          </motion.p>
        </div>

        {/* Architecture diagram */}
        <motion.div
          variants={scaleUpVariants}
          className="bg-white rounded-2xl sm:rounded-3xl shadow-xl p-4 sm:p-8 md:p-12 max-w-4xl mx-auto"
        >
          <div className="space-y-4">
            {/* Input */}
            <motion.div
              className="text-center p-3 sm:p-4 bg-bg-secondary rounded-lg sm:rounded-xl"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="w-8 h-8 sm:w-10 sm:h-10 mx-auto bg-accent-primary/20 rounded-lg flex items-center justify-center">
                <svg className="w-4 h-4 sm:w-5 sm:h-5 text-accent-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <p className="font-medium text-text-primary mt-2 text-xs sm:text-sm lg:text-base">Input: 30 frames × 75 landmarks × 3 coordinates</p>
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
                  className="architecture-box rounded-lg sm:rounded-xl p-4 sm:p-6"
                  style={{ borderLeftColor: comp.color, borderLeftWidth: "4px" }}
                  initial={{ opacity: 0, x: i % 2 === 0 ? -30 : 30 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 + i * 0.15 }}
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="flex justify-between items-start flex-wrap gap-2 sm:gap-4">
                    <div className="min-w-0 flex-1">
                      <h4 className="font-bold text-accent-primary text-base sm:text-lg">{comp.name}</h4>
                      <p className="text-text-secondary text-xs sm:text-sm">{comp.full}</p>
                    </div>
                    <span className="text-[10px] sm:text-xs bg-bg-tertiary px-2 sm:px-3 py-0.5 sm:py-1 rounded-full text-text-secondary font-mono flex-shrink-0">
                      {comp.params}
                    </span>
                  </div>
                  <p className="mt-2 sm:mt-3 text-text-primary text-sm sm:text-base">{comp.desc}</p>
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
              className="text-center p-3 sm:p-4 bg-accent-primary/10 rounded-lg sm:rounded-xl border-2 border-accent-primary"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 1.1 }}
            >
              <div className="w-8 h-8 sm:w-10 sm:h-10 mx-auto bg-accent-primary rounded-lg flex items-center justify-center">
                <svg className="w-4 h-4 sm:w-5 sm:h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="font-medium text-accent-primary mt-2 text-xs sm:text-sm lg:text-base">Output: Sign + Component Scores + Actionable Feedback</p>
            </motion.div>
          </div>
        </motion.div>

        {/* Key insight callout */}
        <motion.div
          variants={fadeUpVariants}
          className="mt-8 sm:mt-12 max-w-3xl mx-auto bg-accent-primary text-white rounded-xl sm:rounded-2xl p-6 sm:p-8 lg:p-10 text-center"
        >
          <p className="text-xs sm:text-sm uppercase tracking-widest mb-2 sm:mb-4 opacity-80">Key Insight</p>
          <p className="text-lg sm:text-xl lg:text-2xl font-medium leading-relaxed">
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
    <Section className="py-16 sm:py-24 lg:py-32 bg-bg-primary relative overflow-hidden" id="results">
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

      <div className="max-w-7xl mx-auto px-4 sm:px-6 relative z-10">
        <div className="text-center max-w-3xl mx-auto mb-10 sm:mb-16 lg:mb-20">
          <motion.p
            variants={fadeUpVariants}
            className="text-accent-primary font-semibold tracking-widest text-xs sm:text-sm uppercase mb-2 sm:mb-4"
          >
            Benchmark Results
          </motion.p>
          <motion.h2
            variants={fadeUpVariants}
            className="font-display text-2xl sm:text-3xl lg:text-h1 text-text-primary mb-4 sm:mb-6"
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
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 lg:gap-6 mb-10 sm:mb-16 lg:mb-20">
          {stats.map((stat, i) => (
            <motion.div
              key={i}
              variants={fadeUpVariants}
              className="relative bg-white rounded-xl sm:rounded-2xl p-4 sm:p-6 lg:p-8 text-center shadow-sm card-hover overflow-hidden"
            >
              {/* Mini animated bar inside card */}
              <motion.div
                className="absolute bottom-0 left-0 h-1 bg-accent-primary/20"
                initial={{ width: "0%" }}
                whileInView={{ width: `${stat.value}%` }}
                transition={{ duration: 1.5, delay: i * 0.15, ease: "easeOut" }}
                viewport={{ once: true }}
              />
              <div className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-bold text-accent-primary stat-number">
                <AnimatedCounter target={stat.value} suffix={stat.suffix} />
              </div>
              <div className="text-text-secondary mt-2 sm:mt-3 font-medium text-xs sm:text-sm">{stat.label}</div>
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
          className="mt-10 sm:mt-16 lg:mt-20 grid md:grid-cols-2 gap-8 sm:gap-10 lg:gap-12 items-center"
        >
          <div>
            <h3 className="font-display text-xl sm:text-2xl lg:text-3xl text-text-primary mb-3 sm:mb-4">
              75 Landmarks Per Frame
            </h3>
            <p className="text-text-secondary text-sm sm:text-base lg:text-lg leading-relaxed mb-4 sm:mb-6">
              MediaPipe extracts precise 3D coordinates from your webcam feed — pose, hands, and fingertips.
              Our models analyze these landmarks in real-time to understand exactly what you're signing.
            </p>
            <ul className="space-y-2 sm:space-y-3">
              <li className="flex items-center gap-2 sm:gap-3 text-text-secondary text-sm sm:text-base">
                <span className="w-2 h-2 bg-accent-primary rounded-full flex-shrink-0"></span>
                33 pose landmarks for body position
              </li>
              <li className="flex items-center gap-2 sm:gap-3 text-text-secondary text-sm sm:text-base">
                <span className="w-2 h-2 bg-accent-secondary rounded-full flex-shrink-0"></span>
                21 landmarks per hand (42 total)
              </li>
              <li className="flex items-center gap-2 sm:gap-3 text-text-secondary text-sm sm:text-base">
                <span className="w-2 h-2 bg-accent-tertiary rounded-full flex-shrink-0"></span>
                Fingertips tracked for precision feedback
              </li>
            </ul>
          </div>
          {/* Video demo without skeleton overlay */}
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="rounded-2xl overflow-hidden shadow-xl bg-text-primary"
          >
            <div className="relative aspect-[4/3]">
              <video
                src="/videos/signs/good.mp4"
                autoPlay
                loop
                muted
                playsInline
                className="w-full h-full object-cover"
              />
              {/* Prediction overlay */}
              <div className="absolute bottom-2 sm:bottom-4 left-2 sm:left-4 right-2 sm:right-4">
                <div className="bg-black/80 backdrop-blur-sm rounded-lg sm:rounded-xl p-2 sm:p-4">
                  <div className="flex items-center justify-between mb-1.5 sm:mb-2">
                    <span className="text-white font-bold text-sm sm:text-lg">GOOD</span>
                    <span className="text-accent-primary font-bold text-sm sm:text-base">93%</span>
                  </div>
                  <div className="grid grid-cols-4 gap-1 sm:gap-2 text-[8px] sm:text-xs">
                    <div>
                      <div className="text-text-tertiary truncate">Handshape</div>
                      <div className="h-1 sm:h-1.5 bg-white/20 rounded mt-0.5 sm:mt-1">
                        <div className="h-full bg-green-500 rounded" style={{ width: '95%' }}></div>
                      </div>
                    </div>
                    <div>
                      <div className="text-text-tertiary truncate">Location</div>
                      <div className="h-1 sm:h-1.5 bg-white/20 rounded mt-0.5 sm:mt-1">
                        <div className="h-full bg-green-500 rounded" style={{ width: '91%' }}></div>
                      </div>
                    </div>
                    <div>
                      <div className="text-text-tertiary truncate">Movement</div>
                      <div className="h-1 sm:h-1.5 bg-white/20 rounded mt-0.5 sm:mt-1">
                        <div className="h-full bg-green-500 rounded" style={{ width: '94%' }}></div>
                      </div>
                    </div>
                    <div>
                      <div className="text-text-tertiary truncate">Orientation</div>
                      <div className="h-1 sm:h-1.5 bg-white/20 rounded mt-0.5 sm:mt-1">
                        <div className="h-full bg-green-500 rounded" style={{ width: '92%' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              {/* Live indicator */}
              <div className="absolute top-2 sm:top-4 left-2 sm:left-4 flex items-center gap-1.5 sm:gap-2 bg-black/60 backdrop-blur-sm rounded-full px-2 sm:px-3 py-1 sm:py-1.5">
                <span className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-red-500 rounded-full animate-pulse"></span>
                <span className="text-white text-[10px] sm:text-xs font-medium">Real-time Analysis</span>
              </div>
            </div>
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
    <Section className="py-16 sm:py-24 lg:py-32 bg-bg-secondary" id="applications">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="text-center max-w-3xl mx-auto mb-10 sm:mb-16 lg:mb-20 px-2">
          <motion.span
            variants={fadeUpVariants}
            className="inline-block text-accent-primary font-semibold tracking-widest text-xs sm:text-sm uppercase mb-2 sm:mb-4"
          >
            Real-World Impact
          </motion.span>
          <motion.h2
            variants={fadeUpVariants}
            className="font-display text-2xl sm:text-3xl lg:text-h1 text-text-primary mb-4 sm:mb-6"
          >
            From Self-Study to{" "}
            <span className="text-gradient">Professional Training</span>
          </motion.h2>
          <motion.p
            variants={fadeUpVariants}
            className="text-text-secondary text-sm sm:text-base lg:text-lg"
          >
            SignSense adapts to diverse learning contexts, providing personalized feedback wherever sign language education happens.
          </motion.p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 lg:gap-8">
          {applications.map((app, i) => (
            <motion.div
              key={i}
              variants={fadeUpVariants}
              className="group relative rounded-xl sm:rounded-2xl overflow-hidden shadow-lg card-hover aspect-[3/2]"
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
              <div className="absolute inset-0 p-4 sm:p-6 flex flex-col justify-end">
                <h3 className="font-display text-lg sm:text-xl lg:text-2xl text-white mb-1 sm:mb-2 group-hover:text-accent-secondary transition-colors">
                  {app.title}
                </h3>
                <p className="text-white/90 text-xs sm:text-sm leading-relaxed opacity-100 sm:opacity-0 group-hover:opacity-100 transition-opacity duration-300">
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
    <Section className="py-16 sm:py-24 lg:py-32 bg-bg-primary" id="demo">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="text-center max-w-3xl mx-auto mb-8 sm:mb-12 lg:mb-16 px-2">
          <motion.span
            variants={fadeUpVariants}
            className="inline-block text-accent-primary font-semibold tracking-widest text-xs sm:text-sm uppercase mb-2 sm:mb-4"
          >
            See It In Action
          </motion.span>
          <motion.h2
            variants={fadeUpVariants}
            className="font-display text-2xl sm:text-3xl lg:text-h1 text-text-primary mb-4 sm:mb-6"
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
          <div className="relative aspect-video rounded-xl sm:rounded-2xl lg:rounded-3xl overflow-hidden shadow-2xl">
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
                className="w-16 h-16 sm:w-20 sm:h-20 lg:w-24 lg:h-24 rounded-full bg-accent-primary/90 flex items-center justify-center mb-4 sm:mb-6 cursor-pointer shadow-xl"
                whileHover={{ scale: 1.1, boxShadow: "0 25px 50px rgba(45, 90, 74, 0.4)" }}
                whileTap={{ scale: 0.95 }}
                animate={{ scale: [1, 1.05, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <svg className="w-8 h-8 sm:w-10 sm:h-10 lg:w-12 lg:h-12 text-white ml-0.5 sm:ml-1" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </motion.div>
              <p className="text-lg sm:text-xl lg:text-2xl font-display font-medium">Watch Demo</p>
              <p className="text-white/70 mt-1 sm:mt-2 text-sm sm:text-base">See SignSense in action</p>
            </div>

            {/* Decorative elements */}
            <div className="absolute bottom-2 sm:bottom-4 left-2 sm:left-4 right-2 sm:right-4 flex justify-between items-center text-white/60 text-xs sm:text-sm">
              <span>0:00</span>
              <div className="flex-1 mx-2 sm:mx-4 h-0.5 sm:h-1 bg-white/20 rounded-full overflow-hidden">
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
            className="mt-4 sm:mt-6 lg:mt-8 flex flex-wrap justify-center gap-2 sm:gap-3"
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
                className="px-3 py-1.5 sm:px-4 sm:py-2 bg-white rounded-full shadow-sm border border-bg-tertiary text-text-secondary text-xs sm:text-sm hover:bg-accent-primary hover:text-white hover:border-accent-primary cursor-pointer transition-all flex items-center gap-1.5 sm:gap-2"
                whileHover={{ scale: 1.05, y: -2 }}
              >
                <span className="text-[10px] sm:text-xs text-text-tertiary">{chapter.time}</span>
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
    <Section className="py-16 sm:py-24 lg:py-32 bg-bg-secondary">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="grid lg:grid-cols-2 gap-8 sm:gap-10 lg:gap-12">
          {/* Specs table */}
          <motion.div variants={fadeUpVariants}>
            <h3 className="font-display text-xl sm:text-2xl lg:text-h2 text-text-primary mb-4 sm:mb-6 lg:mb-8">Technical Specifications</h3>
            <div className="bg-white rounded-xl sm:rounded-2xl overflow-hidden shadow-sm overflow-x-auto">
              <table className="w-full min-w-[300px]">
                <thead className="bg-bg-tertiary">
                  <tr>
                    <th className="px-3 sm:px-6 py-2 sm:py-3 text-left text-xs sm:text-sm font-semibold text-text-primary">Model</th>
                    <th className="px-3 sm:px-6 py-2 sm:py-3 text-right text-xs sm:text-sm font-semibold text-text-primary">Parameters</th>
                    <th className="px-3 sm:px-6 py-2 sm:py-3 text-right text-xs sm:text-sm font-semibold text-text-primary">Latency</th>
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
                      <td className="px-3 sm:px-6 py-3 sm:py-4 font-medium text-text-primary text-xs sm:text-sm">{row.model}</td>
                      <td className="px-3 sm:px-6 py-3 sm:py-4 text-right text-text-secondary font-mono text-xs sm:text-sm">{row.params}</td>
                      <td className="px-3 sm:px-6 py-3 sm:py-4 text-right text-text-secondary font-mono text-xs sm:text-sm">{row.latency}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>

          {/* Privacy callout */}
          <motion.div variants={fadeUpVariants}>
            <h3 className="font-display text-xl sm:text-2xl lg:text-h2 text-text-primary mb-4 sm:mb-6 lg:mb-8">Privacy by Design</h3>
            <div className="bg-accent-primary text-white rounded-xl sm:rounded-2xl p-5 sm:p-6 lg:p-8 space-y-4 sm:space-y-5">
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
                  className="flex items-start gap-3 sm:gap-4"
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  viewport={{ once: true }}
                >
                  <div className="w-8 h-8 sm:w-10 sm:h-10 bg-white/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <svg className="w-4 h-4 sm:w-5 sm:h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      {item.icon}
                    </svg>
                  </div>
                  <div>
                    <h4 className="font-semibold text-base sm:text-lg">{item.title}</h4>
                    <p className="text-white/80 mt-0.5 sm:mt-1 text-sm sm:text-base">{item.desc}</p>
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
    <footer className="bg-text-primary text-white py-10 sm:py-12 lg:py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 sm:gap-8 lg:gap-12 mb-8 sm:mb-12">
          <div className="col-span-2">
            <div className="flex items-center gap-3 mb-3 sm:mb-4">
              <span className="font-display text-2xl sm:text-3xl text-white">SignSense</span>
            </div>
            <p className="text-white/60 max-w-md leading-relaxed text-sm sm:text-base">
              AI-powered sign language learning platform using four specialized neural networks
              for real-time, component-specific feedback.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-3 sm:mb-4 text-sm sm:text-base">Research</h4>
            <ul className="space-y-1.5 sm:space-y-2 text-white/60 text-sm">
              <li><a href="#" className="hover:text-white transition-colors">Paper (PDF)</a></li>
              <li><a href="#" className="hover:text-white transition-colors">GitHub</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Benchmarks</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-3 sm:mb-4 text-sm sm:text-base">Resources</h4>
            <ul className="space-y-1.5 sm:space-y-2 text-white/60 text-sm">
              <li><a href="#" className="hover:text-white transition-colors">Documentation</a></li>
              <li><a href="#" className="hover:text-white transition-colors">API Reference</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Tutorials</a></li>
            </ul>
          </div>
        </div>
        <div className="border-t border-white/10 pt-6 sm:pt-8 flex flex-col sm:flex-row justify-between items-center gap-3 sm:gap-4 text-white/40 text-xs sm:text-sm">
          <p>© 2026 SignSense. Built for Toshiba Challenge.</p>
          <div className="flex gap-4 sm:gap-6">
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
