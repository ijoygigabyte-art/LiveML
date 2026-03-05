/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0E1117',
        surface: '#1A1F2E',
        border: '#2D3348',
        primary: '#FF6B6B',
        accent: '#FFE66D'
      }
    },
  },
  plugins: [],
}
