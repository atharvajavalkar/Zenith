/**
 * Zenith mark + wordmark
 * Mark: twin peaks to a gold point — height, precision, conviction.
 */
export default function ZenithLogo({
  size = 28,
  showText = true,
  showTagline = false,
  variant = 'dark',
  animated = false,
  textClassName = 'text-[1.35rem]',
  className = '',
}) {
  const isLight = variant === 'light'
  const stroke = isLight ? '#6e5535' : '#c9a66b'

  return (
    <div className={`flex items-center gap-3.5 ${className}`}>
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 64 64"
        width={size}
        height={size}
        className="shrink-0"
        aria-hidden="true"
      >
        <g stroke={stroke} strokeLinecap="round" strokeLinejoin="round" fill="none">
          <path d="M12 48 L32 12 L52 48" strokeWidth="1.75" />
          <path d="M20 48 L32 24 L44 48" strokeWidth="1.1" opacity="0.55" />
        </g>
        <circle cx="32" cy="12" r="2.35" fill={stroke} />
        <path
          d="M16 52 H48"
          stroke={stroke}
          strokeWidth="1"
          strokeLinecap="round"
          opacity="0.35"
          fill="none"
        />
      </svg>
      {showText && (
        <div>
          <span
            className={`brand-zenith block ${textClassName} ${
              isLight ? 'brand-zenith--light' : animated ? 'brand-zenith--auth' : ''
            }`}
          >
            Zenith
          </span>
          {showTagline && (
            <p
              className={`text-[9px] tracking-[0.22em] uppercase mt-1.5 font-light ${
                isLight ? 'text-[#787B86]' : 'text-surface-muted'
              }`}
            >
              Peak of gold
            </p>
          )}
        </div>
      )}
    </div>
  )
}
