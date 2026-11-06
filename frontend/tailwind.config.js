/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#080807',
          raised: '#0e0e0c',
          card: '#121210',
          cardHover: '#171714',
          elevated: '#1a1a17',
          border: '#242420',
          muted: '#8a857c',
        },
        ink: {
          DEFAULT: '#ece8e1',
          soft: '#c9c3b8',
          faint: '#8a857c',
        },
        accent: {
          // Club emerald — clear on black, old-money not neon
          buy: '#2f9e6f',
          'buy-deep': '#1f7350',
          // Oxblood — decisive, not muddy salmon
          sell: '#d45b5b',
          'sell-deep': '#a63d3d',
          gold: '#c9a66b',
          'gold-soft': '#dcc08a',
          'gold-deep': '#9a7a4f',
          'gold-line': 'rgba(201, 166, 107, 0.22)',
        },
      },
      fontFamily: {
        sans: ['Outfit', 'ui-sans-serif', 'sans-serif'],
        display: ['"Cormorant Garamond"', 'Georgia', 'serif'],
      },
      letterSpacing: {
        brand: '0.28em',
        nav: '0.14em',
        label: '0.18em',
      },
      boxShadow: {
        soft: '0 24px 60px rgba(0,0,0,0.35)',
        hairline: 'inset 0 0 0 1px rgba(196,165,116,0.08)',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-soft': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.45' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.55s ease-out both',
        'pulse-soft': 'pulse-soft 2.4s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
