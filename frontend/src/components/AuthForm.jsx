import { Link } from 'react-router-dom'
import ZenithLogo from './ZenithLogo'

function AuthLayout({ title, subtitle, children }) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-surface px-4">
      <div className="w-full max-w-md animate-fade-in-up">
        <div className="flex flex-col items-center mb-8">
          <ZenithLogo size={72} textClassName="text-5xl sm:text-6xl" className="justify-center" />
          <p className="text-surface-muted mt-4 tracking-wide">{subtitle}</p>
        </div>
        <div className="zcard p-8 overflow-hidden">
          <div className="gold-rule mb-6 -mt-2" />
          <h2 className="text-xl font-semibold mb-6 text-gray-100">{title}</h2>
          {children}
        </div>
      </div>
    </div>
  )
}

export function AuthForm({ onSubmit, children, submitLabel, error, footer }) {
  return (
    <form onSubmit={onSubmit} className="space-y-4">
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg px-4 py-3 text-sm">
          {error}
        </div>
      )}
      {children}
      <button
        type="submit"
        className="w-full bg-gold-gradient text-surface font-semibold py-2.5 rounded-lg transition hover:shadow-glow hover:brightness-105"
      >
        {submitLabel}
      </button>
      {footer}
    </form>
  )
}

export function AuthField({ label, type, value, onChange, placeholder }) {
  return (
    <div>
      <label className="block text-sm text-gray-400 mb-1">{label}</label>
      <input
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required
        className="w-full bg-surface border border-surface-border rounded-lg px-4 py-2.5 text-white transition focus:outline-none focus:border-accent-gold focus:ring-2 focus:ring-accent-gold/20"
      />
    </div>
  )
}

export { AuthLayout, Link }
