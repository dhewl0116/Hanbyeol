/** @type {import('tailwindcss').Config} */
export const content = [
  "./src/**/*.{js,jsx,ts,tsx}", // src 디렉토리의 모든 JS, JSX, TS, TSX 파일을 대상으로 TailwindCSS 적용
];
export const theme = {
  extend: {
    colors: {
      primary: "#4571e5",
      secondary: "#e7e9f5",
      fontgrey: "#353535",
      fontlight: "#00000080",
    },
  },
};
export const plugins = [];
