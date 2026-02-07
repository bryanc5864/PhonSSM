import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SignSense | AI-Powered Sign Language Learning",
  description: "Learn American Sign Language with real-time AI feedback. Four specialized neural networks work together to give you component-specific corrections and accelerate your learning.",
  keywords: ["sign language", "ASL", "AI", "machine learning", "accessibility", "education"],
  authors: [{ name: "SignSense Team" }],
  icons: {
    icon: "/favicon.svg",
  },
  openGraph: {
    title: "SignSense | AI-Powered Sign Language Learning",
    description: "Learn ASL with real-time AI feedback from four specialized neural networks.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
