// Interactive three.js viewer for a benchmark's .geo geometry (benchmarks and leaderboard pages).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const COLORS = { surface: 0xeef5ff, outline: 0x283d60, mesh: 0x7fa6dc, hover: 0x075de1 };
// Muted colours for named (physical) surfaces, assigned in file order.
const GROUP_PALETTE = [0x6f95cc, 0xd39a6f, 0x7fb394, 0xa995d1, 0xbcc6d4, 0xcf8a97, 0x8fbcc2];
const OPACITY = { plane: 0.22, side: 0.5 };
// Everything is drawn in a fixed order without depth testing: the faces are translucent,
// and tiny z offsets would z-fight and flicker once the view is rotated.
const LAYERS = { surface: 0, mesh: 1, outline: 2, marker: 3 };

const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[char]);
const toCss = (hex) => `#${hex.toString(16).padStart(6, "0")}`;

document.querySelectorAll("[data-geometry-viewer]").forEach(init);

function flat(object, layer) {
  object.renderOrder = layer;
  object.material.depthTest = false;
  object.material.depthWrite = false;
  return object;
}

function init(card) {
  const stage = card.querySelector("[data-geometry-stage]");
  const status = card.querySelector("[data-geometry-status]");
  const meshButton = card.querySelector("[data-geometry-mesh]");
  const resetButton = card.querySelector("[data-geometry-reset]");
  const datasetFilter = document.querySelector("#dataset-filter");
  const { formatLength } = window.FluidBenchGeo;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const tooltip = document.createElement("div");
  tooltip.className = "geometry-tooltip";
  tooltip.hidden = true;
  stage.append(renderer.domElement, tooltip);
  const legend = document.createElement("ul");
  legend.className = "geometry-legend";
  stage.after(legend);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(12, 2, 0.1, 5000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.rotateSpeed = 0.5;
  controls.addEventListener("change", render);
  controls.addEventListener("start", () => { userMoved = true; dragging = true; hideHover(); });
  controls.addEventListener("end", () => { dragging = false; });

  let root = null;
  let geometry = null;
  let meshLines = null;
  let currentPath = null;
  let userMoved = false;
  let dragging = false;
  let groupMaterials = new Map();
  let pickables = [];
  let vertices = [];
  let marker = null;
  const raycaster = new THREE.Raycaster();

  function render() {
    renderer.render(scene, camera);
  }

  function resize() {
    // clientWidth/Height are unzoomed CSS pixels (the site sets `zoom` on <html>).
    const { clientWidth: width, clientHeight: height } = stage;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    render();
  }
  new ResizeObserver(() => { resize(); if (geometry && !userMoved) resetView(); }).observe(stage);

  // Default view: looking straight down the z axis with the whole domain in frame.
  function resetView() {
    if (!geometry) return;
    const { bounds } = geometry;
    const margin = 0.03 * Math.max(bounds.width, bounds.height);
    const halfWidth = bounds.width / 2 + margin;
    const halfHeight = bounds.height / 2 + margin;
    const vFov = THREE.MathUtils.degToRad(camera.fov);
    const distance = Math.max(halfHeight, halfWidth / camera.aspect) / Math.tan(vFov / 2);
    controls.target.set(0, 0, 0);
    camera.position.set(0, 0, distance);
    camera.up.set(0, 1, 0);
    controls.update();
    userMoved = false;
    render();
  }

  const segmentsFrom = (pairs, color, layer) => {
    const buffer = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pairs, 3));
    return flat(new THREE.LineSegments(buffer, new THREE.LineBasicMaterial({ color })), layer);
  };

  function planeFace(loop, offset, circles, material) {
    const shape = new THREE.Shape(loop.map((p) => new THREE.Vector2(p.x, p.y)));
    // Round obstacles are cut out of the fill so they read as holes from any angle.
    circles.forEach((c) => shape.holes.push(new THREE.Path().absarc(c.x, c.y, c.radius, 0, 2 * Math.PI, true)));
    const mesh = new THREE.Mesh(new THREE.ShapeGeometry(shape, 24), material);
    mesh.position.set(offset.x, offset.y, offset.z);
    return mesh;
  }

  function sideFace(polyline, vector, material) {
    const positions = [];
    for (let k = 0; k + 1 < polyline.length; k += 1) {
      const a = polyline[k], b = polyline[k + 1];
      const a2 = [a.x + vector.x, a.y + vector.y, vector.z], b2 = [b.x + vector.x, b.y + vector.y, vector.z];
      positions.push(a.x, a.y, 0, b.x, b.y, 0, ...b2, a.x, a.y, 0, ...b2, ...a2);
    }
    const buffer = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    return new THREE.Mesh(buffer, material);
  }

  function build(geo) {
    const { bounds, surfaces, curves, circles, groups, extrusion } = geo;
    const vector = extrusion?.vector || { x: 0, y: 0, z: 0 };
    const group = new THREE.Group();
    group.position.set(-(bounds.minX + bounds.maxX) / 2, -(bounds.minY + bounds.maxY) / 2, -vector.z / 2);
    groupMaterials = new Map();
    pickables = [];

    const colorOf = new Map(groups.map((g, index) => [g.name, GROUP_PALETTE[index % GROUP_PALETTE.length]]));
    if (groups.length) {
      groups.forEach((g) => {
        const materials = [];
        g.faces.forEach((face) => {
          const opacity = face.type === "plane" ? OPACITY.plane : OPACITY.side;
          const material = new THREE.MeshBasicMaterial({ color: colorOf.get(g.name), transparent: true, opacity, side: THREE.DoubleSide });
          material.userData.baseOpacity = opacity;
          const mesh = face.type === "plane" ? planeFace(face.loop, face.offset, circles, material) : sideFace(face.polyline, face.vector, material);
          mesh.userData = { kind: "surface", group: g.name, z: face.type === "plane" ? face.offset.z : null };
          materials.push(material);
          pickables.push(mesh);
          group.add(flat(mesh, LAYERS.surface));
        });
        groupMaterials.set(g.name, materials);
      });
    } else {
      surfaces.forEach((loop) => {
        const fill = planeFace(loop, { x: 0, y: 0, z: 0 }, circles, new THREE.MeshBasicMaterial({ color: COLORS.surface, side: THREE.DoubleSide }));
        fill.userData = { kind: "surface", group: null, z: 0 };
        pickables.push(fill);
        group.add(flat(fill, LAYERS.surface));
      });
    }

    // Edges: each curve at the base (and the extruded copy), coloured by its named surface.
    const offsets = vector.z || vector.x || vector.y ? [{ x: 0, y: 0, z: 0 }, vector] : [{ x: 0, y: 0, z: 0 }];
    curves.forEach((curve) => {
      const base = new THREE.Color(curve.group && colorOf.has(curve.group) ? colorOf.get(curve.group) : COLORS.outline);
      const color = curve.group && colorOf.has(curve.group) ? base.lerp(new THREE.Color(COLORS.outline), 0.35) : base;
      offsets.forEach((offset, copy) => {
        const points = curve.polyline.map((p) => new THREE.Vector3(p.x + offset.x, p.y + offset.y, offset.z));
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color }));
        line.userData = { kind: "curve", id: copy ? null : curve.id, group: curve.group, type: curve.type };
        pickables.push(line);
        group.add(flat(line, LAYERS.outline));
      });
    });
    vertices = geo.points.flatMap((p) => offsets.map((offset, copy) => ({
      id: copy ? null : p.id, x: p.x + offset.x, y: p.y + offset.y, z: p.z + offset.z,
    })));
    if (offsets.length > 1) {
      const pairs = geo.points.filter((p) => p.onCurve).flatMap((p) => [p.x, p.y, 0, p.x + vector.x, p.y + vector.y, vector.z]);
      group.add(segmentsFrom(pairs, COLORS.outline, LAYERS.outline));
    }

    const size = Math.max(bounds.width, bounds.height);
    marker = flat(new THREE.Mesh(new THREE.SphereGeometry(0.011 * size, 16, 10), new THREE.MeshBasicMaterial({ color: COLORS.hover })), LAYERS.marker);
    marker.visible = false;
    group.add(marker);

    legend.innerHTML = groups.map((g) => `<li data-group="${escapeHtml(g.name)}"><span class="geometry-swatch" style="background:${toCss(colorOf.get(g.name))}"></span>${escapeHtml(g.name)}</li>`).join("");
    legend.hidden = !groups.length;
    return group;
  }

  function highlight(name) {
    groupMaterials.forEach((materials, groupName) => materials.forEach((material) => {
      const base = material.userData.baseOpacity;
      material.opacity = name == null ? base : groupName === name ? Math.min(0.8, base * 1.8) : base * 0.35;
    }));
    legend.querySelectorAll("li").forEach((item) => item.classList.toggle("is-active", item.dataset.group === name));
    render();
  }

  function buildMesh(geo) {
    const grids = geo.grids();
    if (!grids.length) return null;
    const vector = geo.extrusion?.vector;
    const pairs = [];
    grids.forEach(({ nu, nv, nodes }) => {
      for (let j = 0; j < nv; j += 1) {
        for (let i = 0; i < nu; i += 1) {
          const p = nodes[j][i];
          if (i + 1 < nu) pairs.push(p.x, p.y, 0, nodes[j][i + 1].x, nodes[j][i + 1].y, 0);
          if (j + 1 < nv) pairs.push(p.x, p.y, 0, nodes[j + 1][i].x, nodes[j + 1][i].y, 0);
        }
      }
    });
    if (vector) {
      // One extruded layer: copy the grid onto the top face and join the boundary nodes.
      const base = pairs.length;
      for (let k = 0; k < base; k += 3) pairs.push(pairs[k] + vector.x, pairs[k + 1] + vector.y, vector.z);
      grids.forEach(({ nu, nv, nodes }) => nodes.forEach((row, j) => row.forEach((p, i) => {
        if (i === 0 || j === 0 || i === nu - 1 || j === nv - 1) pairs.push(p.x, p.y, 0, p.x + vector.x, p.y + vector.y, vector.z);
      })));
    }
    return { lines: segmentsFrom(pairs, COLORS.mesh, LAYERS.mesh), grids };
  }

  function setMeshVisible(visible) {
    if (visible && !meshLines && geometry) {
      meshLines = buildMesh(geometry);
      if (meshLines) root.add(meshLines.lines);
    }
    if (meshLines) meshLines.lines.visible = visible;
    if (meshButton) {
      meshButton.setAttribute("aria-pressed", String(visible));
      meshButton.textContent = visible ? "Hide mesh" : "Mesh it";
    }
    render();
  }

  // Hover read-out in the spirit of Gmsh's status bar: the entity under the cursor and its coordinates.
  const xyz = (p) => `(${[p.x, p.y, p.z].map((value) => formatLength(value)).join(", ")})`;

  function hideHover() {
    tooltip.hidden = true;
    if (marker?.visible) { marker.visible = false; highlight(null); }
  }

  function describeHover(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    const sx = event.clientX - rect.left;
    const sy = event.clientY - rect.top;
    const pointer = new THREE.Vector2((sx / rect.width) * 2 - 1, -(sy / rect.height) * 2 + 1);
    const pixel = 9 / rect.width; // snapping radius in normalised device units (x)

    // 1. Geometry points within a few pixels.
    let best = null;
    vertices.forEach((v) => {
      const projected = root.localToWorld(new THREE.Vector3(v.x, v.y, v.z)).project(camera);
      const distance = Math.hypot((projected.x - pointer.x) * camera.aspect, projected.y - pointer.y);
      if (distance < pixel * 2 * camera.aspect && (!best || distance < best.distance)) best = { distance, v };
    });
    if (best) {
      const { v } = best;
      return { at: v, text: `<strong>Point${v.id == null ? "" : ` ${v.id}`}</strong> ${xyz(v)}` };
    }

    // 2. Mesh nodes, curves and surfaces from a ray cast.
    raycaster.setFromCamera(pointer, camera);
    const size = Math.max(geometry.bounds.width, geometry.bounds.height);
    raycaster.params.Line.threshold = 0.008 * size;
    const hits = raycaster.intersectObjects(pickables, false);
    if (!hits.length) return null;
    const curveHit = hits.find((hit) => hit.object.userData.kind === "curve");
    const hit = curveHit || hits[0];
    const local = root.worldToLocal(hit.point.clone());
    const info = hit.object.userData;

    if (meshLines?.lines.visible && info.kind === "surface") {
      let nearest = null;
      meshLines.grids.forEach(({ nodes }) => nodes.forEach((row, j) => row.forEach((p, i) => {
        const distance = Math.hypot(p.x - local.x, p.y - local.y);
        if (!nearest || distance < nearest.distance) nearest = { distance, p, i, j };
      })));
      if (nearest) {
        const vector = geometry.extrusion?.vector;
        const z = vector && local.z > vector.z / 2 ? vector.z : 0;
        const node = { x: nearest.p.x, y: nearest.p.y, z };
        return { at: node, group: info.group, text: `<strong>Node</strong> (${nearest.i}, ${nearest.j}) ${xyz(node)}${info.group ? `<br>${escapeHtml(info.group)}` : ""}` };
      }
    }
    const kind = info.kind === "curve" ? `Curve${info.id == null ? "" : ` ${info.id}`}` : "Surface";
    return { at: local, group: info.group, text: `<strong>${kind}</strong>${info.group ? ` · ${escapeHtml(info.group)}` : ""}<br>${xyz(local)}` };
  }

  renderer.domElement.addEventListener("pointermove", (event) => {
    if (!geometry || dragging) return;
    const hover = describeHover(event);
    if (!hover) { hideHover(); return; }
    tooltip.innerHTML = hover.text;
    tooltip.hidden = false;
    const rect = renderer.domElement.getBoundingClientRect();
    const scale = stage.clientWidth / rect.width; // undo the page zoom
    const x = (event.clientX - rect.left) * scale;
    const y = (event.clientY - rect.top) * scale;
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
    tooltip.classList.toggle("is-left", x > stage.clientWidth * 0.6);
    marker.position.set(hover.at.x, hover.at.y, hover.at.z);
    marker.visible = true;
    highlight(hover.group ?? null);
  });
  renderer.domElement.addEventListener("pointerleave", hideHover);
  legend.addEventListener("pointerover", (event) => {
    const item = event.target.closest("li[data-group]");
    if (item) highlight(item.dataset.group);
  });
  legend.addEventListener("pointerleave", () => highlight(null));

  async function show(dataset) {
    const path = dataset?.geometry || "";
    if (path === currentPath) return;
    currentPath = path;
    if (root) scene.remove(root);
    root = null;
    geometry = null;
    meshLines = null;
    hideHover();
    card.hidden = !path;
    if (!path) return;

    status.textContent = "Loading geometry…";
    try {
      const loaded = await window.FluidBenchGeo.load(path);
      if (path !== currentPath) return;
      geometry = loaded;
      root = build(geometry);
      scene.add(root);
      status.textContent = "";
      if (meshButton) meshButton.disabled = !geometry.grids().length;
      setMeshVisible(false);
      resize();
      resetView();
    } catch (error) {
      status.textContent = error.message;
    }
  }

  meshButton?.addEventListener("click", () => setMeshVisible(meshButton.getAttribute("aria-pressed") !== "true"));
  resetButton?.addEventListener("click", resetView);

  window.FluidBenchData.loadBenchmarks().then((datasets) => {
    // Leaderboard viewers follow the benchmark filter; benchmark-page viewers follow their data-dataset-id.
    const current = () => {
      const id = card.hasAttribute("data-follow-selection") ? datasetFilter?.value : card.dataset.datasetId;
      return datasets.find((item) => item.id === id) || datasets[0];
    };
    show(current());
    document.addEventListener("fluidbench:benchmark-selected", () => show(current()));
  }).catch((error) => { status.textContent = error.message; });
}
