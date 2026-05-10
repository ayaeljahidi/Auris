import { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';

const vertexShader = `
  varying vec3 vWorldPosition;
  varying vec3 vNormal;
  uniform float uTime;
  uniform vec3 uSunPosition;
  uniform float uScale;
  uniform vec3 uClick;

  float terrainHeight(vec3 p) {
    float t = uTime;
    float s = uScale;
    float y = 0.0;
    y += sin((p.x * 1.0 / s) + t * 0.5) * 1.0;
    y += sin((p.z * 1.0 / s) + t * 0.5) * 1.0;
    y += sin((p.x * 2.0 / s) + t * 1.0 + p.z) * 0.5;
    y += sin((p.z * 2.0 / s) + t * 1.1) * 0.5;
    y += sin((p.x * 4.0 / s) + t * 2.0) * 0.25;
    y += sin((p.z * 4.0 / s) + t * 2.1) * 0.25;
    y += sin((p.x * 8.0 / s) + t * 4.0) * 0.125;
    y += sin((p.z * 8.0 / s) + t * 4.1) * 0.125;
    y += sin((p.x * 16.0 / s) + t * 8.0) * 0.0625;
    y += sin((p.z * 16.0 / s) + t * 8.1) * 0.0625;
    y += sin((p.x * 32.0 / s) + t * 16.0) * 0.03125;
    y += sin((p.z * 32.0 / s) + t * 16.1) * 0.03125;
    float clickDist = length(p.xz - uClick.xy);
    float clickWave = sin(clickDist * 0.15 - t * 3.0) * exp(-clickDist * 0.05) * uClick.z;
    y += clickWave * 1.8;
    return y;
  }

  void main() {
    vec3 pos = position;
    float z = terrainHeight(pos);
    vWorldPosition = pos + vec3(0.0, z, 0.0);
    float eps = 0.01;
    float hX = terrainHeight(pos + vec3(eps, 0.0, 0.0)) - terrainHeight(pos - vec3(eps, 0.0, 0.0));
    float hZ = terrainHeight(pos + vec3(0.0, 0.0, eps)) - terrainHeight(pos - vec3(0.0, 0.0, eps));
    vNormal = normalize(vec3(-hX, 2.0 * eps, -hZ));
    pos.y += z;
    vec4 modelViewPosition = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * modelViewPosition;
  }
`;

const fragmentShader = `
  varying vec3 vWorldPosition;
  varying vec3 vNormal;
  uniform float uTime;
  uniform vec3 uSunPosition;
  uniform vec3 uColor[6];

  float terrainHeight(vec3 p) {
    float t = uTime;
    float s = 20.0;
    float y = 0.0;
    y += sin((p.x * 1.0 / s) + t * 0.5) * 1.0;
    y += sin((p.z * 1.0 / s) + t * 0.5) * 1.0;
    y += sin((p.x * 2.0 / s) + t * 1.0 + p.z) * 0.5;
    y += sin((p.z * 2.0 / s) + t * 1.1) * 0.5;
    y += sin((p.x * 4.0 / s) + t * 2.0) * 0.25;
    y += sin((p.z * 4.0 / s) + t * 2.1) * 0.25;
    y += sin((p.x * 8.0 / s) + t * 4.0) * 0.125;
    y += sin((p.z * 8.0 / s) + t * 4.1) * 0.125;
    y += sin((p.x * 16.0 / s) + t * 8.0) * 0.0625;
    y += sin((p.z * 16.0 / s) + t * 8.1) * 0.0625;
    y += sin((p.x * 32.0 / s) + t * 16.0) * 0.03125;
    y += sin((p.z * 32.0 / s) + t * 16.1) * 0.03125;
    return y;
  }

  void main() {
    vec2 wp = vWorldPosition.xz;
    float t = uTime;
    float h = terrainHeight(vWorldPosition);
    float hx = terrainHeight(vWorldPosition + vec3(1.0, 0.0, 0.0));
    float hy_v = terrainHeight(vWorldPosition + vec3(0.0, 0.0, 1.0));
    float h2 = terrainHeight(vWorldPosition + vec3(-1.0, 0.0, -1.0));
    vec3 noise = vec3(h, hx, hy_v) - h2;
    noise = normalize(noise);
    vec3 lightDirection = normalize(uSunPosition - vWorldPosition);
    float lightIntensity = max(dot(vNormal, lightDirection), 0.0);
    vec4 finalColor = vec4(0.0);

    vec4 deepWater = vec4(uColor[0], 1.0);
    float mask = smoothstep(0.0, 0.8, h) * (1.0 - smoothstep(0.8, 1.0, h));
    finalColor += deepWater * mask;

    vec4 water = vec4(uColor[1], 1.0);
    mask = smoothstep(0.4, 1.2, h) * (1.0 - smoothstep(1.2, 2.0, h));
    finalColor += water * mask;

    vec4 shore = vec4(uColor[2], 1.0);
    mask = smoothstep(1.0, 2.5, h) * (1.0 - smoothstep(2.5, 4.0, h));
    finalColor += shore * mask;

    vec4 vegetation = vec4(uColor[3], 1.0);
    mask = smoothstep(2.0, 4.0, h) * (1.0 - smoothstep(4.0, 6.0, h));
    finalColor += vegetation * mask;

    vec4 rock = vec4(uColor[4], 1.0);
    mask = smoothstep(4.0, 6.0, h) * (1.0 - smoothstep(6.0, 8.0, h));
    finalColor += rock * mask;

    vec4 snow = vec4(uColor[5], 1.0);
    mask = smoothstep(6.5, 8.5, h);
    finalColor += snow * mask;

    finalColor.rgb += (noise * 0.1) + (h * 0.05);
    float fogDepth = length(vWorldPosition.xyz - cameraPosition);
    vec3 fogColor = vec3(0.05, 0.1, 0.2);
    float fogNear = 0.0;
    float fogFar = 150.0;
    float fogFactor = smoothstep(fogNear, fogFar, fogDepth);
    finalColor.rgb = mix(finalColor.rgb, fogColor, fogFactor);
    vec3 viewDirection = normalize(cameraPosition - vWorldPosition.xyz);
    float rimLight = pow(1.0 - max(dot(viewDirection, vNormal), 0.0), 3.0);
    vec3 sunColor = vec3(1.0, 0.9, 0.8);
    finalColor.rgb = mix(finalColor.rgb, sunColor, rimLight * 0.5);
    gl_FragColor = finalColor;
  }
`;

const skyVertexShader = `
  varying vec3 vNormal;
  varying vec3 vPosition;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vPosition = (modelMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const skyFragmentShader = `
  varying vec3 vNormal;
  varying vec3 vPosition;
  uniform vec3 uColor;
  uniform float uIntensity;
  void main() {
    vec3 viewDirection = normalize(vPosition - cameraPosition);
    float fresnel = 1.0 - dot(viewDirection, vNormal);
    gl_FragColor = vec4(uColor, fresnel * uIntensity);
  }
`;

export default function HeroTerrain() {
  const containerRef = useRef<HTMLDivElement>(null);
  const mouseRef = useRef({ x: 0, y: 0 });
  const clickRef = useRef({ x: 0, y: 0, z: 0, intensity: 0 });
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const frameRef = useRef<number>(0);

  const colorPalette = useMemo(() => [
    new THREE.Vector3(0.04, 0.1, 0.22),
    new THREE.Vector3(0.08, 0.18, 0.35),
    new THREE.Vector3(0.15, 0.28, 0.45),
    new THREE.Vector3(0.25, 0.4, 0.55),
    new THREE.Vector3(0.4, 0.55, 0.7),
    new THREE.Vector3(0.6, 0.7, 0.85),
  ], []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(15, 8, 20);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setClearColor(0x0e1a33);
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.display = 'block';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Terrain
    const geometry = new THREE.PlaneGeometry(60, 60, 128, 128);
    const terrainMaterial = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uSunPosition: { value: new THREE.Vector3(15, 20, 20) },
        uScale: { value: 20 },
        uClick: { value: new THREE.Vector3(0, 0, 0) },
        uColor: { value: colorPalette },
      },
      side: THREE.DoubleSide,
    });
    const terrain = new THREE.Mesh(geometry, terrainMaterial);
    terrain.rotation.x = -Math.PI / 2;
    terrain.scale.set(0.8, 0.8, 0.8);
    scene.add(terrain);

    // Sky dome
    const skyGeometry = new THREE.SphereGeometry(200, 32, 32);
    const skyMaterial = new THREE.ShaderMaterial({
      vertexShader: skyVertexShader,
      fragmentShader: skyFragmentShader,
      uniforms: {
        uColor: { value: new THREE.Vector3(0.04, 0.1, 0.22) },
        uIntensity: { value: 0.8 },
      },
      transparent: true,
      side: THREE.BackSide,
      depthWrite: false,
    });
    const sky = new THREE.Mesh(skyGeometry, skyMaterial);
    scene.add(sky);

    // Raycaster for click
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const planeIntersect = new THREE.Vector3();

    const clock = new THREE.Clock();
    const lookAtTarget = new THREE.Vector3(0, 0, 0);

    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      terrainMaterial.uniforms.uTime.value = elapsed;

      // Mouse camera parallax
      const targetX = mouseRef.current.x * 15;
      const targetY = mouseRef.current.y * 8 + 8;
      camera.position.x += (targetX - camera.position.x) * 0.05;
      camera.position.y += (targetY - camera.position.y) * 0.05;
      camera.lookAt(lookAtTarget);

      // Click decay
      if (clickRef.current.intensity > 0.01) {
        clickRef.current.intensity *= 0.96;
        terrainMaterial.uniforms.uClick.value.set(
          clickRef.current.x,
          clickRef.current.y,
          clickRef.current.intensity
        );
      } else {
        terrainMaterial.uniforms.uClick.value.set(0, 0, 0);
        clickRef.current.intensity = 0;
      }

      renderer.render(scene, camera);
    };
    animate();

    // Events
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouseRef.current.y = -(e.clientY / window.innerHeight) * 2 + 1;
    };

    const handleClick = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      raycaster.ray.intersectPlane(plane, planeIntersect);
      if (planeIntersect) {
        clickRef.current.x = planeIntersect.x;
        clickRef.current.y = planeIntersect.z;
        clickRef.current.intensity = 1.0;
      }
    };

    const handleResize = () => {
      if (!container) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };

    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    container.addEventListener('click', handleClick);
    window.addEventListener('resize', handleResize, { passive: true });

    return () => {
      cancelAnimationFrame(frameRef.current);
      window.removeEventListener('mousemove', handleMouseMove);
      container.removeEventListener('click', handleClick);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      geometry.dispose();
      terrainMaterial.dispose();
      skyGeometry.dispose();
      skyMaterial.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [colorPalette]);

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 z-0"
      role="presentation"
      aria-hidden="true"
    />
  );
}
