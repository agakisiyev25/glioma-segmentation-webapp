import { lazy, Suspense, useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";

import { getInfo } from "./lib/api";
import type { ApiInfo } from "./lib/types";
import About from "./pages/About";
import Upload from "./pages/Upload";

// The viewer pulls in three.js; load it only when a result is opened.
const Viewer = lazy(() => import("./pages/Viewer"));

function lastJob(): string | null {
  try {
    return window.localStorage.getItem("glioma.lastJob");
  } catch {
    return null;
  }
}

function ModeBadge({ info }: { info: ApiInfo | null }) {
  if (!info) return null;
  if (info.mode === "demo") return <span className="mode-badge demo" title={info.message}>Demo mode</span>;
  if (info.mode === "model_error") return <span className="mode-badge error" title={info.message}>Model error</span>;
  return (
    <span className="mode-badge ok" title={info.message}>
      Model loaded · {info.model?.device?.toUpperCase() ?? ""}
    </span>
  );
}

export default function App() {
  const location = useLocation(); // re-read the last job when the route changes
  const recent = lastJob();
  const [info, setInfo] = useState<ApiInfo | null>(null);
  useEffect(() => {
    getInfo().then(setInfo).catch(() => setInfo(null));
  }, []);

  return (
    <div className="app" data-route={location.pathname.startsWith("/viewer") ? "viewer" : "page"}>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          Postoperative glioma segmentation
          <span className="brand-tag">research prototype</span>
        </div>
        <nav>
          <ModeBadge info={info} />
          <NavLink to="/" end>
            New case
          </NavLink>
          {recent && <NavLink to={`/viewer/${recent}`}>Last result</NavLink>}
          <NavLink to="/about">About</NavLink>
        </nav>
      </header>
      <Suspense fallback={<main className="viewer viewer-message"><div className="loading-card"><div className="spinner" aria-hidden="true" /><ol><li className="current">Loading the viewer</li></ol></div></main>}>
        <Routes>
          <Route path="/" element={<Upload />} />
          <Route path="/viewer/:jobId" element={<Viewer />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Upload />} />
        </Routes>
      </Suspense>
    </div>
  );
}
