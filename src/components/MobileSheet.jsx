// A bottom sheet that slides up over the canvas on mobile.
export default function MobileSheet({ title, onClose, children, height = '64vh' }) {
  return (
    <div className="m-sheet-wrap" onClick={onClose}>
      <div className="m-sheet" style={{ maxHeight: height }} onClick={(e) => e.stopPropagation()}>
        <div className="m-sheet-head">
          <span>{title}</span>
          <button className="x" onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="m-sheet-body scrolly">{children}</div>
      </div>
    </div>
  )
}
