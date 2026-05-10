import { useEffect, useRef, useCallback } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const AMPLITUDE = 25;
const SPRING_STIFFNESS = 0.08;
const NUM_LINES = 8;
const LINE_GAP = 10;

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function generateColors(count: number) {
  const colors = [];
  for (let i = 0; i < count; i++) {
    const t = i / (count - 1);
    const r = Math.round(255 + (121 - 255) * t);
    const g = Math.round(107 + (147 - 107) * t);
    const b = Math.round(53 + (30 - 53) * t);
    colors.push(`rgb(${r}, ${g}, ${b})`);
  }
  return colors;
}

interface Point {
  x: number;
  y: number;
}

class WaveLine {
  y: number;
  baseY: number;
  points: Point[];
  phase: number;
  color: string;
  amplitude: number;
  targetAmplitude: number;
  dragOffset: number;
  targetDragOffset: number;
  totalPoints: number;

  constructor(baseY: number, color: string) {
    this.y = baseY;
    this.baseY = baseY;
    this.points = [];
    this.phase = Math.random() * Math.PI * 2;
    this.color = color;
    this.amplitude = 0;
    this.targetAmplitude = AMPLITUDE;
    this.dragOffset = 0;
    this.targetDragOffset = 0;
    this.totalPoints = 0;
  }

  generate(width: number) {
    this.points = [];
    this.totalPoints = Math.floor(width / 5) + 2;
    for (let i = 0; i <= this.totalPoints; i++) {
      this.points.push({ x: i * 5, y: 0 });
    }
  }

  update(
    time: number,
    mouseX: number,
    mouseY: number,
    mouseSpeed: number,
    isPressed: boolean,
    width: number,
    _height: number
  ) {
    this.amplitude = lerp(this.amplitude, this.targetAmplitude, 0.05);
    this.dragOffset = lerp(this.dragOffset, this.targetDragOffset, SPRING_STIFFNESS);
    this.y = lerp(this.y, this.baseY + this.dragOffset, 0.12);

    const distToMouse = mouseX >= 0 ? Math.abs(this.y - mouseY) : undefined;

    for (let i = 0; i < this.points.length; i++) {
      const pt = this.points[i];
      const nx = pt.x / width;
      pt.y = 0;
      pt.y += Math.sin(nx * Math.PI * 2 + time * 0.002 + this.phase) * this.amplitude;
      pt.y += Math.sin(nx * Math.PI * 4 - time * 0.003) * (this.amplitude * 0.5);
      pt.y += Math.cos(nx * Math.PI * 3 + time * 0.0015) * (this.amplitude * 0.3);

      if (mouseX >= 0 && distToMouse !== undefined && distToMouse < 250) {
        const mouseInfluence = Math.pow(1 - distToMouse / 250, 2);
        pt.y += Math.sin(nx * Math.PI * 10 + time * 0.01) * mouseInfluence * 15;
        pt.y += Math.cos(nx * Math.PI * 8 - time * 0.008) * mouseInfluence * 10;
      }

      if (mouseSpeed > 2) {
        const speedInfluence = Math.min(mouseSpeed / 20, 1);
        pt.y += Math.sin(nx * Math.PI * 6 + time * 0.02) * speedInfluence * 20;
      }

      if (isPressed && mouseX >= 0) {
        const distToClickX = Math.abs(pt.x - mouseX);
        if (distToClickX < 200) {
          const influence = Math.pow(1 - distToClickX / 200, 2);
          pt.y += Math.sin(nx * Math.PI * 2 + time * 0.01) * influence * 30;
        }
      }
    }
  }

  draw(ctx: CanvasRenderingContext2D, _width: number) {
    ctx.beginPath();
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    if (this.points.length === 0) return;
    ctx.moveTo(this.points[0].x, this.y + this.points[0].y);
    for (let i = 1; i < this.points.length; i++) {
      const prev = this.points[i - 1];
      const curr = this.points[i];
      const cpx = (prev.x + curr.x) / 2;
      const cpy = (this.y + prev.y + this.y + curr.y) / 2;
      ctx.quadraticCurveTo(cpx, cpy, curr.x, this.y + curr.y);
    }
    ctx.stroke();
  }
}

export default function WaveformBanner() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const linesRef = useRef<WaveLine[]>([]);
  const mouseRef = useRef({ x: -1, y: -1, speed: 0, lastX: -1, isDragging: false });
  const frameRef = useRef(0);

  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const time = Date.now();
    const mouse = mouseRef.current;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    linesRef.current.forEach((line) => {
      line.update(time, mouse.x, mouse.y, mouse.speed, mouse.isDragging, canvas.width, canvas.height);
      line.draw(ctx, canvas.width);
    });

    if (!mouse.isDragging) {
      mouse.speed *= 0.9;
      if (mouse.speed < 0.1) {
        mouse.x = -1;
        mouse.y = -1;
        mouse.speed = 0;
      }
    }

    frameRef.current = requestAnimationFrame(animate);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio, 2);
      canvas.width = container.clientWidth * dpr;
      canvas.height = container.clientHeight * dpr;
      canvas.style.width = container.clientWidth + 'px';
      canvas.style.height = container.clientHeight + 'px';
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.scale(dpr, dpr);

      const colors = generateColors(NUM_LINES);
      const w = container.clientWidth;
      const h = container.clientHeight;
      linesRef.current = [];
      for (let i = 0; i < NUM_LINES; i++) {
        const baseY = h / 2 - ((NUM_LINES - 1) * LINE_GAP) / 2 + i * LINE_GAP;
        const line = new WaveLine(baseY, colors[i]);
        line.generate(w);
        linesRef.current.push(line);
      }
    };

    resize();
    frameRef.current = requestAnimationFrame(animate);

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      if (mouseRef.current.isDragging && mouseRef.current.lastX >= 0) {
        const deltaX = x - mouseRef.current.lastX;
        mouseRef.current.speed = Math.abs(deltaX);
        linesRef.current.forEach((line) => {
          line.targetDragOffset = Math.max(-200, Math.min(200, line.targetDragOffset - deltaX));
        });
      }
      mouseRef.current.x = x;
      mouseRef.current.y = y;
      mouseRef.current.lastX = x;
    };

    const handleMouseDown = (e: MouseEvent) => {
      mouseRef.current.isDragging = true;
      const rect = canvas.getBoundingClientRect();
      mouseRef.current.lastX = e.clientX - rect.left;
    };

    const handleMouseUp = () => {
      mouseRef.current.isDragging = false;
      mouseRef.current.lastX = -1;
      linesRef.current.forEach((line) => {
        line.targetDragOffset = 0;
      });
    };

    const handleMouseLeave = () => {
      mouseRef.current.isDragging = false;
      mouseRef.current.lastX = -1;
      linesRef.current.forEach((line) => {
        line.targetDragOffset = 0;
      });
    };

    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseLeave);
    window.addEventListener('resize', resize, { passive: true });

    return () => {
      cancelAnimationFrame(frameRef.current);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mousedown', handleMouseDown);
      canvas.removeEventListener('mouseup', handleMouseUp);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
      window.removeEventListener('resize', resize);
    };
  }, [animate]);

  // Entrance animation
  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.fromTo(
        containerRef.current,
        { opacity: 0, scale: 0.98 },
        {
          opacity: 1,
          scale: 1,
          duration: 1,
          ease: 'power2.out',
          scrollTrigger: { trigger: containerRef.current, start: 'top 80%' },
        }
      );
    });
    return () => ctx.revert();
  }, []);

  return (
    <section className="w-full py-[clamp(80px,12vh,140px)] px-6 bg-auris-bg">
      <div
        ref={containerRef}
        className="relative w-full h-[400px] overflow-hidden rounded-none"
        style={{ background: 'linear-gradient(to bottom, #0A0F1A 0%, #111827 50%, #0A0F1A 100%)' }}
      >
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full cursor-grab active:cursor-grabbing"
        />
        <div className="absolute inset-0 flex flex-col items-center justify-center z-10 pointer-events-none">
          <div className="flex items-center gap-2 mb-3">
            <span className="w-2 h-2 rounded-full bg-auris-teal shadow-[0_0_10px_#14B8A6] animate-pulse-dot" />
            <span className="label-tag text-auris-teal">LIVE ANALYSIS</span>
          </div>
          <h2 className="font-mono text-mono-display text-auris-text mb-2">Feel the rhythm</h2>
          <p className="text-body-sm text-auris-text-secondary max-w-[400px] text-center mb-4">
            Drag across the waveform to explore how AI analyzes speech patterns in real time.
          </p>
          <span className="text-caption text-auris-text-tertiary flex items-center gap-1">
            Click and drag <span className="animate-bounce-x">→</span>
          </span>
        </div>
      </div>
    </section>
  );
}
