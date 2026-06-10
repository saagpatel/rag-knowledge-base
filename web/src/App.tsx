import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import DashboardPage from "./pages/DashboardPage";
import SearchPage from "./pages/SearchPage";
import AskPage from "./pages/AskPage";
import CollectionsPage from "./pages/CollectionsPage";
import DocumentsPage from "./pages/DocumentsPage";
import IngestPage from "./pages/IngestPage";
import AnalyticsPage from "./pages/AnalyticsPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route path="search" element={<SearchPage />} />
          <Route path="ask" element={<AskPage />} />
          <Route path="collections" element={<CollectionsPage />} />
          <Route path="documents" element={<DocumentsPage />} />
          <Route path="ingest" element={<IngestPage />} />
          <Route path="analytics" element={<AnalyticsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
