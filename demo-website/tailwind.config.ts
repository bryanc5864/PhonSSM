import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Warm neutrals
        bg: {
          primary: "#FDFCFB",
          secondary: "#F7F5F3",
          tertiary: "#EDEAE6",
        },
        // Rich text colors
        text: {
          primary: "#1A1614",
          secondary: "#5C5552",
          tertiary: "#8A8583",
        },
        // Distinctive accents
        accent: {
          primary: "#2D5A4A",
          secondary: "#E8B86D",
          tertiary: "#C75D4D",
          light: "#3D7A6A",
        },
        success: "#3D8B6E",
        warning: "#D4A84B",
        error: "#C45C4C",
      },
      fontFamily: {
        display: ["'Instrument Serif'", "Georgia", "serif"],
        body: ["'Inter'", "-apple-system", "sans-serif"],
        mono: ["'JetBrains Mono'", "'SF Mono'", "monospace"],
      },
      fontSize: {
        hero: ["clamp(3rem, 8vw, 6rem)", { lineHeight: "1.1" }],
        h1: ["clamp(2.5rem, 5vw, 4rem)", { lineHeight: "1.15" }],
        h2: ["clamp(1.75rem, 3vw, 2.5rem)", { lineHeight: "1.2" }],
        h3: ["clamp(1.25rem, 2vw, 1.5rem)", { lineHeight: "1.3" }],
      },
      animation: {
        "fade-up": "fadeUp 0.6s ease forwards",
        "fade-in": "fadeIn 0.6s ease forwards",
        "slide-up": "slideUp 0.8s ease forwards",
        "draw-line": "drawLine 1s ease forwards",
        "pulse-soft": "pulseSoft 2s ease-in-out infinite",
        "float": "float 6s ease-in-out infinite",
        "gradient": "gradient 8s ease infinite",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(30px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(60px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        drawLine: {
          "0%": { strokeDashoffset: "1000" },
          "100%": { strokeDashoffset: "0" },
        },
        pulseSoft: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.7" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-20px)" },
        },
        gradient: {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "hero-gradient": "linear-gradient(135deg, #FDFCFB 0%, #F7F5F3 50%, #EDEAE6 100%)",
        "accent-gradient": "linear-gradient(135deg, #2D5A4A 0%, #3D7A6A 100%)",
      },
    },
  },
  plugins: [],
};
export default config;
