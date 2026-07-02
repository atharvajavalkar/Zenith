/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#0a0e14',
          card: '#131b27',
          cardHover: '#18222f',
          elevated: '#1b2634',
          border: '#26344a',
          muted: '#8b97a8',
        },
        accent: {
          buy: '#22c55e',
          sell: '#ef4444',
          gold: '#f3b94a',
          'gold-soft': '#e7c067',
          'gold-deep': '#c5862f',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      boxShadow: {
        card: '0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -16px rgba(0,0,0,0.7)',
        'card-hover': '0 1px 2px rgba(0,0,0,0.4), 0 18px 40px -18px rgba(0,0,0,0.8)',
        glow: '0 0 0 1px rgba(243,185,74,0.18), 0 10px 34px -10px rgba(243,185,74,0.28)',
      },
      backgroundImage: {
        'gold-gradient': 'linear-gradient(135deg, #f3b94a 0%, #e7c067 45%, #c5862f 100%)',
      },
      keyframes: {
        'fade-in-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-soft': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.45' },
        },
      },
      animation: {
        'fade-in-up': 'fade-in-up 0.45s cubic-bezier(0.16, 1, 0.3, 1) both',
        'pulse-soft': 'pulse-soft 2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
