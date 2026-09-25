import React, { useRef, useEffect } from 'react';

// Deterministic vibrant color generator for classes
const CLASS_COLORS = [
  '#38bdf8', // sky blue
  '#4ade80', // green
  '#f43f5e', // rose
  '#a855f7', // purple
  '#fbbf24', // amber
  '#2dd4bf', // teal
  '#f97316', // orange
  '#ec4899', // pink
  '#6366f1', // indigo
  '#84cc16', // lime
];

function getClassColor(className) {
  let hash = 0;
  for (let i = 0; i < className.length; i++) {
    hash = (hash << 5) - hash + className.charCodeAt(i);
  }
  const idx = Math.abs(hash) % CLASS_COLORS.length;
  return CLASS_COLORS[idx];
}

export function BoundingBoxCanvas({
  imageSrc = null,
  detections = [],
  tracks = [],
  width = 640,
  height = 480,
  showLabels = true,
  showConfidence = true,
  showTracks = true,
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // If imageSrc is present, load and draw image
    if (imageSrc) {
      const img = new window.Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        canvas.width = img.naturalWidth || width;
        canvas.height = img.naturalHeight || height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        drawOverlays(ctx, canvas.width, canvas.height);
      };
      img.src = imageSrc;
    } else {
      // Just clear canvas or draw empty background
      canvas.width = width;
      canvas.height = height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawOverlays(ctx, width, height);
    }
  }, [imageSrc, detections, tracks, width, height, showLabels, showConfidence, showTracks]);

  const drawOverlays = (ctx, cWidth, cHeight) => {
    // Prefer tracks if available and showTracks is true; else fall back to detections
    const items = (showTracks && tracks && tracks.length > 0) ? tracks : (detections || []);

    items.forEach((item) => {
      // Support both { box: { x1, y1, x2, y2 } } and { x1, y1, x2, y2 }
      const box = item.box || item;
      const x1 = Math.max(0, box.x1);
      const y1 = Math.max(0, box.y1);
      const x2 = Math.min(cWidth, box.x2);
      const y2 = Math.min(cHeight, box.y2);
      const bWidth = x2 - x1;
      const bHeight = y2 - y1;

      if (bWidth <= 0 || bHeight <= 0) return;

      const className = item.class_name || 'object';
      const color = getClassColor(className);
      const trackId = item.track_id !== undefined ? item.track_id : null;
      const conf = item.confidence !== undefined ? Math.round(item.confidence * 100) : null;

      // 1. Draw bounding box rectangle
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1, y1, bWidth, bHeight);

      // 2. Build text tag
      let tag = className;
      if (trackId !== null) {
        tag = `#${trackId} ${tag}`;
      }
      if (showConfidence && conf !== null) {
        tag += ` ${conf}%`;
      }

      if (showLabels) {
        ctx.font = 'bold 12px Inter, sans-serif';
        const textMetrics = ctx.measureText(tag);
        const textW = textMetrics.width;
        const textH = 16;
        const tagY = Math.max(0, y1 - textH);

        // Tag background
        ctx.fillStyle = color;
        ctx.fillRect(x1, tagY, textW + 8, textH);

        // Tag text
        ctx.fillStyle = '#0f172a';
        ctx.fillText(tag, x1 + 4, tagY + 12);
      }
    });
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <canvas
        ref={canvasRef}
        style={{
          maxWidth: '100%',
          maxHeight: '100%',
          objectFit: 'contain',
          borderRadius: '8px',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)',
        }}
      />
    </div>
  );
}
