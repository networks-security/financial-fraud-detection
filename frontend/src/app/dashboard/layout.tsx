import { Grid } from "antd";
import SideNavigation from "./components/SideNavigation";
import dashboardStyles from "./dashboard.module.scss";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      <SideNavigation />
      <div style={{ marginLeft: 200 }}>{children}</div>
    </>
  );
}
