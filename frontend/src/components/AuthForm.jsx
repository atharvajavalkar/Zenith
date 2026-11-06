import ZenithLogo from './ZenithLogo'

function AuthLayout({ title, subtitle, children }) {
  return (
    <div className="app-shell min-h-screen flex items-center justify-center px-5">
      <div className="relative z-[1] w-full max-w-md animate-fade-in">
        <div className="flex flex-col items-center mb-10 text-center">
          <ZenithLogo
            size={42}
            animated
            showTagline
            textClassName="text-4xl sm:text-5xl"
            className="justify-center"
          />
          <p className="text-surface-muted mt-5 text-sm font-light tracking-wide">{subtitle}</p>
        </div>
        <div className="zcard p-8 sm:p-9">
          <h2 className="font-display text-2xl font-medium tracking-[0.04em] text-ink mb-6">{title}</h2>
          {children}
        </div>
      </div>
    </div>
  )
}

export function AuthForm({ onSubmit, children, submitLabel, error, footer }) {
  return (
    <form onSubmit={onSubmit} className="space-y-5">
      {error && (
        <div className="border border-accent-sell/30 bg-accent-sell/10 text-accent-sell rounded-sm px-4 py-3 text-sm font-light">
          {error}
        </div>
      )}
      {children}
      <button type="submit" className="btn-primary w-full py-3 mt-2">
        {submitLabel}
      </button>
      {footer}
    </form>
  )
}

export function AuthField({ label, type, value, onChange, placeholder }) {
  return (
    <div>
      <label className="block text-[10px] uppercase tracking-label text-surface-muted mb-2">{label}</label>
      <input
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required
        className="field-input"
      />
    </div>
  )
}

export { AuthLayout }
