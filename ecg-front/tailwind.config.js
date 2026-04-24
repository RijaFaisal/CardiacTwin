// Configured for zinc dark theme
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Tailwind's native 'zinc' handles the grayish-dark perfectly,
                // but you can add specific medical alert colors here if needed.
            }
        },
    },
    plugins: [],
}