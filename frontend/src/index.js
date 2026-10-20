import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

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

reportWebVitals();
