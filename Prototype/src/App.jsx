import React, { useState } from "react";
import { C } from "./theme.js";
import Sidebar from "./components/Sidebar.jsx";
import TopBar from "./components/TopBar.jsx";
import CausalStrip from "./components/CausalStrip.jsx";

import OverviewPage from "./pages/Overview.jsx";
import ExperimentsPage from "./pages/Experiments.jsx";
import CreateExperimentPage from "./pages/CreateExperiment.jsx";
import LiveTrainingPage from "./pages/LiveTraining.jsx";
import EnvironmentsPage from "./pages/Environments.jsx";
import FailuresPage from "./pages/Failures.jsx";
import CurriculumPage from "./pages/Curriculum.jsx";
import EvaluationPage from "./pages/Evaluation.jsx";
import ComparisonPage from "./pages/Comparison.jsx";
import DetailsPage from "./pages/Details.jsx";

export default function App() {
  const [page, setPage] = useState("overview");
  const pages = {
    overview: <OverviewPage setPage={setPage} />,
    experiments: <ExperimentsPage setPage={setPage} />,
    create: <CreateExperimentPage setPage={setPage} />,
    live: <LiveTrainingPage />,
    environments: <EnvironmentsPage />,
    failures: <FailuresPage />,
    curriculum: <CurriculumPage />,
    evaluation: <EvaluationPage />,
    comparison: <ComparisonPage />,
    details: <DetailsPage />,
  };
  return (
    <div className="flex w-full min-h-screen font-sans" style={{ background: C.bg, color: C.text }}>
      <Sidebar page={page} setPage={setPage} />
      <div className="flex-1 flex flex-col min-w-0">
        <TopBar page={page} />
        <CausalStrip />
        <div className="overflow-y-auto">{pages[page]}</div>
      </div>
    </div>
  );
}
