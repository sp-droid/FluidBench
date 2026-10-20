// Interactive three.js viewer for a benchmark's mesh (assets/meshes/*.json, written by benchmarks/mesh_export.py;
// the format is documented there). Benchmarks and leaderboard pages.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const COLORS = { surface: 0xeef5ff, outline: 0x283d60, mesh: 0x7fa6dc, hover: 0x075de1 };
// Muted colours for named boundaries, assigned in file order.
const GROUP_PALETTE = [0x6f95cc, 0xd39a6f, 0x7fb394, 0xa995d1, 0xbcc6d4, 0xcf8a97, 0x8fbcc2];
const OPACITY = { plane: 0.22, side: 0.5 };
// Everything is drawn in a fixed order without depth testing: the faces are translucent,
// and tiny z offsets would z-fight and flicker once the view is rotated.
const LAYERS = { surface: 0, mesh: 1, outline: 2, marker: 3 };

const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[char]);
const toCss = (hex) => `#${hex.toString(16).padStart(6, "0")}`;
const formatLength = (value) => Number(value.toFixed(3)).toString();

const cache = new Map();
function loadMesh(path) {
  const url = window.FluidBenchPaths.url(path);
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then((response) => {
      if (!response.ok) throw new Error(`Could not load ${url} (${response.status}).`);
      return response.json();
    }).then(describe));
  }
  return cache.get(url);
}

// Derived drawing data: bounds, unique cell edges and the positions of the data nodes.
function describe(mesh) {
  if (mesh.schemaVersion !== 1) throw new Error("Unsupported mesh file.");
  const { vertices, cells, cellSize } = mesh;
  const vertexCount = vertices.length / 2;
  const bounds = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
  for (let k = 0; k < vertices.length; k += 2) {
    bounds.minX = Math.min(bounds.minX, vertices[k]);
    bounds.maxX = Math.max(bounds.maxX, vertices[k]);
    bounds.minY = Math.min(bounds.minY, vertices[k + 1]);
    bounds.maxY = Math.max(bounds.maxY, vertices[k + 1]);
  }
  bounds.width = bounds.maxX - bounds.minX;
  bounds.height = bounds.maxY - bounds.minY;

  const seen = new Set();
  const edges = [];
  for (let c = 0; c < cells.length; c += cellSize) {
    for (let k = 0; k < cellSize; k += 1) {
      const a = cells[c + k], b = cells[c + (k + 1) % cellSize];
      const key = Math.min(a, b) * vertexCount + Math.max(a, b);
      if (!seen.has(key)) { seen.add(key); edges.push(a, b); }
    }
  }

  // Data nodes: grid vertices, or cell centres for finite-volume data.
  let nodeXY = vertices;
  if (mesh.nodes === "cells") {
    nodeXY = new Float64Array((cells.length / cellSize) * 2);
    for (let c = 0; c < cells.length; c += cellSize) {
      for (let k = 0; k < cellSize; k += 1) {
        nodeXY[(c / cellSize) * 2] += vertices[cells[c + k] * 2] / cellSize;
        nodeXY[(c / cellSize) * 2 + 1] += vertices[cells[c + k] * 2 + 1] / cellSize;
      }
    }
  }
  const depth = mesh.extrusion?.depth || 0;
  return { ...mesh, bounds, edges, nodeXY, depth };
}

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

  const bufferOf = (positions) => new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const segmentsFrom = (pairs, color, layer) => flat(new THREE.LineSegments(bufferOf(pairs), new THREE.LineBasicMaterial({ color })), layer);
  const levels = (depth) => (depth ? [0, depth] : [0]);

  // Every cell (fan-triangulated) at height z.
  function cellTriangles(geo, z) {
    const { vertices, cells, cellSize } = geo;
    const xyz = (v) => [vertices[v * 2], vertices[v * 2 + 1], z];
    const positions = [];
    for (let c = 0; c < cells.length; c += cellSize) {
      for (let k = 1; k + 1 < cellSize; k += 1) positions.push(...xyz(cells[c]), ...xyz(cells[c + k]), ...xyz(cells[c + k + 1]));
    }
    return positions;
  }

  // Boundary faces as line pairs at height z.
  function facePairs(geo, faces, z) {
    const { vertices } = geo;
    const pairs = [];
    for (const v of faces) pairs.push(vertices[v * 2], vertices[v * 2 + 1], z);
    return pairs;
  }

  // Lateral quads swept from the boundary faces along the extrusion depth.
  function sideTriangles(geo, faces) {
    const { vertices, depth } = geo;
    const positions = [];
    for (let k = 0; k < faces.length; k += 2) {
      const [ax, ay] = [vertices[faces[k] * 2], vertices[faces[k] * 2 + 1]];
      const [bx, by] = [vertices[faces[k + 1] * 2], vertices[faces[k + 1] * 2 + 1]];
      positions.push(ax, ay, 0, bx, by, 0, bx, by, depth, ax, ay, 0, bx, by, depth, ax, ay, depth);
    }
    return positions;
  }

  // Ends of each boundary's face chains (where one condition meets another): vertical edges of the extruded case.
  function chainEnds(faces) {
    const degree = new Map();
    for (const v of faces) degree.set(v, (degree.get(v) || 0) + 1);
    return [...degree].filter(([, count]) => count === 1).map(([v]) => v);
  }

  function build(geo) {
    const { bounds, depth } = geo;
    const group = new THREE.Group();
    group.position.set(-(bounds.minX + bounds.maxX) / 2, -(bounds.minY + bounds.maxY) / 2, -depth / 2);
    groupMaterials = new Map();
    pickables = [];

    const planeName = depth ? geo.extrusion.name || null : null;
    const names = [...(planeName ? [planeName] : []), ...geo.boundaries.map((b) => b.name)];
    const colorOf = new Map(names.map((name, index) => [name, GROUP_PALETTE[index % GROUP_PALETTE.length]]));
    const addMaterial = (name, material) => {
      if (!groupMaterials.has(name)) groupMaterials.set(name, []);
      groupMaterials.get(name).push(material);
    };

    // Domain fill: the cells themselves, on both planes of an extruded case.
    levels(depth).forEach((z) => {
      const material = planeName
        ? new THREE.MeshBasicMaterial({ color: colorOf.get(planeName), transparent: true, opacity: OPACITY.plane, side: THREE.DoubleSide })
        : new THREE.MeshBasicMaterial({ color: COLORS.surface, side: THREE.DoubleSide });
      material.userData.baseOpacity = material.opacity;
      if (planeName) addMaterial(planeName, material);
      const fill = new THREE.Mesh(bufferOf(cellTriangles(geo, z)), material);
      fill.userData = { kind: "surface", group: planeName, z };
      pickables.push(fill);
      group.add(flat(fill, LAYERS.surface));
    });

    geo.boundaries.forEach(({ name, faces }) => {
      if (depth) {
        const material = new THREE.MeshBasicMaterial({ color: colorOf.get(name), transparent: true, opacity: OPACITY.side, side: THREE.DoubleSide });
        material.userData.baseOpacity = OPACITY.side;
        addMaterial(name, material);
        const side = new THREE.Mesh(bufferOf(sideTriangles(geo, faces)), material);
        side.userData = { kind: "surface", group: name, z: null };
        pickables.push(side);
        group.add(flat(side, LAYERS.surface));
      }
      const color = new THREE.Color(colorOf.get(name)).lerp(new THREE.Color(COLORS.outline), 0.35);
      levels(depth).forEach((z) => {
        const line = segmentsFrom(facePairs(geo, faces, z), color, LAYERS.outline);
        line.userData = { kind: "curve", group: name };
        pickables.push(line);
        group.add(line);
      });
      if (depth) {
        const pairs = chainEnds(faces).flatMap((v) => [geo.vertices[v * 2], geo.vertices[v * 2 + 1], 0, geo.vertices[v * 2], geo.vertices[v * 2 + 1], depth]);
        group.add(segmentsFrom(pairs, COLORS.outline, LAYERS.outline));
      }
    });

    // True geometry under an approximating mesh (e.g. the cylinder behind a masked grid), dashed.
    (geo.reference || []).forEach(({ polyline }) => levels(depth).forEach((z) => {
      const points = [];
      for (let k = 0; k < polyline.length; k += 2) points.push(new THREE.Vector3(polyline[k], polyline[k + 1], z));
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineDashedMaterial({ color: COLORS.outline, dashSize: 0.012 * bounds.width, gapSize: 0.008 * bounds.width }));
      line.computeLineDistances();
      group.add(flat(line, LAYERS.outline));
    }));

    const size = Math.max(bounds.width, bounds.height);
    marker = flat(new THREE.Mesh(new THREE.SphereGeometry(0.011 * size, 16, 10), new THREE.MeshBasicMaterial({ color: COLORS.hover })), LAYERS.marker);
    marker.visible = false;
    group.add(marker);

    legend.innerHTML = names.map((name) => `<li data-group="${escapeHtml(name)}"><span class="geometry-swatch" style="background:${toCss(colorOf.get(name))}"></span>${escapeHtml(name)}</li>`).join("");
    legend.hidden = !names.length;
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

  // Every cell edge, on both planes of an extruded case, joined at the boundary vertices.
  function buildMesh(geo) {
    const { vertices, edges, depth } = geo;
    const pairs = [];
    levels(depth).forEach((z) => {
      for (const v of edges) pairs.push(vertices[v * 2], vertices[v * 2 + 1], z);
    });
    if (depth) {
      const onBoundary = new Set(geo.boundaries.flatMap((b) => b.faces));
      onBoundary.forEach((v) => pairs.push(vertices[v * 2], vertices[v * 2 + 1], 0, vertices[v * 2], vertices[v * 2 + 1], depth));
    }
    return segmentsFrom(pairs, COLORS.mesh, LAYERS.mesh);
  }

  function setMeshVisible(visible) {
    if (visible && !meshLines && geometry) {
      meshLines = buildMesh(geometry);
      root.add(meshLines);
    }
    if (meshLines) meshLines.visible = visible;
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

  // Nearest data node: grid nodes are labelled (i, j), cell-centred nodes by cell index.
  function nearestNode(x, y) {
    const { nodeXY: nodes } = geometry;
    let best = -1;
    let bestDistance = Infinity;
    for (let k = 0; k < nodes.length; k += 2) {
      const distance = (nodes[k] - x) ** 2 + (nodes[k + 1] - y) ** 2;
      if (distance < bestDistance) { bestDistance = distance; best = k / 2; }
    }
    const ny = geometry.grid?.shape?.[1];
    const label = geometry.nodes === "cells" ? `Cell ${best}` : ny ? `Node (${Math.floor(best / ny)}, ${best % ny})` : `Node ${best}`;
    return { x: nodes[best * 2], y: nodes[best * 2 + 1], label };
  }

  function describeHover(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const size = Math.max(geometry.bounds.width, geometry.bounds.height);
    raycaster.params.Line.threshold = 0.008 * size;
    const hits = raycaster.intersectObjects(pickables, false);
    if (!hits.length) return null;
    const hit = hits.find((candidate) => candidate.object.userData.kind === "curve") || hits[0];
    const local = root.worldToLocal(hit.point.clone());
    const info = hit.object.userData;

    if (meshLines?.visible && info.kind === "surface") {
      const node = nearestNode(local.x, local.y);
      const at = { x: node.x, y: node.y, z: info.z ?? (local.z > geometry.depth / 2 ? geometry.depth : 0) };
      return { at, group: info.group, text: `<strong>${node.label}</strong> ${xyz(at)}${info.group ? `<br>${escapeHtml(info.group)}` : ""}` };
    }
    const kind = info.kind === "curve" ? "Boundary" : "Surface";
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
      const loaded = await loadMesh(path);
      if (path !== currentPath) return;
      geometry = loaded;
      root = build(geometry);
      scene.add(root);
      status.textContent = "";
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

document.querySelectorAll("[data-geometry-viewer]").forEach(init);
