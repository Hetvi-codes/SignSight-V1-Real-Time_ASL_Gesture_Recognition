import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

// Suppress benign ResizeObserver notifications from triggering React's development overlay
window.addEventListener('error', (e) => {
  if (
    e.message === 'ResizeObserver loop completed with undelivered notifications.' ||
    e.message === 'ResizeObserver loop limit exceeded' ||
    (typeof e.message === 'string' && e.message.includes('ResizeObserver'))
  ) {
    e.stopImmediatePropagation();
    e.preventDefault();
  }
});

const originalError = console.error;
console.error = (...args) => {
  if (
    typeof args[0] === 'string' &&
    (args[0].includes('ResizeObserver loop completed with undelivered notifications') ||
     args[0].includes('ResizeObserver loop limit exceeded'))
  ) {
    return;
  }
  originalError(...args);
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
