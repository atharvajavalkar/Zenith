/**
 * Zenith gold peak mark + optional wordmark.
 * variant "dark" = default app header; "light" = chart page white header
 */
export default function ZenithLogo({
  size = 40,
  showText = true,
  showTagline = false,
  variant = 'dark',
  textClassName = 'text-4xl',
  className = '',
}) {
  const isLight = variant === 'light'
  const markSize = size

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <img
        src="/logo.svg"
        alt="Zenith — Mont Blanc gold peak"
        width={markSize}
        height={markSize}
        className="shrink-0"
        style={{
          filter: isLight
            ? 'drop-shadow(0 1px 2px rgba(0,0,0,0.08))'
            : 'drop-shadow(0 0 12px rgba(231, 192, 103, 0.42)) drop-shadow(0 2px 6px rgba(0,0,0,0.35))',
        }}
      />
      {showText && (
        <div>
          <span
            className={`brand-zenith block leading-none ${textClassName} ${isLight ? 'brand-zenith--light' : ''}`}
          >
            Zenith
          </span>
          {showTagline && (
            <p
              className={`text-[11px] tracking-[0.16em] uppercase mt-1 ${isLight ? '' : 'text-surface-muted'}`}
              style={isLight ? { color: '#787B86' } : undefined}
            >
              Monte Carlo Signal Engine
            </p>
          )}
        </div>
      )}
    </div>
  )
}
