"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { HorizontalLogo } from "../../shared/components/Logo";
import { APP_COLORS } from "../../shared/utils/app-colors";
// import TransactionsIcon from "../../shared/components/icons/TransactionsIcon";
// import SettingsIcon from "../../shared/components/icons/SettingsIcon";
// import AccountDetailsIcon from "../../shared/components/icons/AccountDetailsIcon";
// import AnalysisIcon from "../../shared/components/icons/AnalysisIcon";
// import NotificationsIcon from "../../shared/components/icons/NotificationsIcon";
import dashboard from "../dashboard.module.scss";

import { Refine } from "@refinedev/core";
import routerProvider from "@refinedev/nextjs-router";
import dataProvider from "@refinedev/simple-rest";
import { Layout, Menu, ConfigProvider } from "antd";
import { useMenu, useLogout } from "@refinedev/core";
import { icons } from "antd/es/image/PreviewGroup";
import {
  SwapOutlined,
  SettingOutlined,
  UserOutlined,
  NotificationOutlined,
  FundOutlined,
} from "@ant-design/icons";

const menuItems = [
  {
    key: "transactions",
    label: "Transactions",
    icon: <SwapOutlined />,
    route: "/dashboard/transactions",
  },
  {
    key: "analysis",
    label: "Analysis",
    icon: <FundOutlined />,
    route: "/dashboard/analysis",
  },
  {
    key: "account-details",
    label: "Account Details",
    icon: <UserOutlined />,
    route: "/dashboard/account-details",
  },
  {
    key: "notifications",
    label: "Notifications",
    icon: <NotificationOutlined />,
    route: "/dashboard/notifications",
  },
  {
    key: "settings",
    label: "Settings",
    icon: <SettingOutlined />,
    route: "/dashboard/settings",
  },
];

export default function SideNavigation() {
  return (
    // Documentation on ConfigProvider
    //https://ant.design/docs/react/customize-theme#customize-design-token
    <div className={dashboard["side-navigation"]}>
      <ConfigProvider
        theme={{
          token: {
            colorPrimary: APP_COLORS.focusPrimary, // Example: Change primary color
            colorPrimaryBg: APP_COLORS.bgLight,
            colorIcon: APP_COLORS.focusPrimary,
            colorIconHover: APP_COLORS.focusSecondary,
            // Other design tokens https://ant.design/docs/react/customize-theme#maptoken
          },
        }}
      >
        <Refine
          routerProvider={routerProvider}
          dataProvider={dataProvider("https://api.fake-rest.refine.dev")}
          resources={[
            {
              name: "transactions",
              list: "/dashboard/transactions",
              meta: {
                label: "Transactions",
                icon: <SwapOutlined />,
                // icon: <TransactionsIcon margin="0px 6px 0px 0px" />,
              },
            },
            {
              name: "analysis",
              list: "/dashboard/analysis",
              meta: {
                label: "Analysis",
                icon: <FundOutlined />,
                // icon: <AnalysisIcon margin="0px 6px 0px 0px" />,
              },
            },
            {
              name: "account-details",
              list: "/dashboard/account-details",
              meta: {
                label: "Account Details",
                icon: <UserOutlined />,
                // icon: <AccountDetailsIcon margin="0px 6px 0px 0px" />,
              },
            },
            {
              name: "notifications",
              list: "/dashboard/notifications",
              meta: {
                label: "Notifications",
                icon: <NotificationOutlined />,
                // icon: <NotificationsIcon margin="0px 6px 0px 0px" />,
              },
            },
            {
              name: "settings",
              list: "/dashboard/settings",
              meta: {
                label: "Settings",
                icon: <SettingOutlined />,
                // icon: <SettingsIcon margin="0px 6px 0px 0px" />,
              },
            },
          ]}
          options={{ syncWithLocation: true }}
        >
          <SideNavigationContents />
        </Refine>
      </ConfigProvider>
    </div>
  );
}

// NOTE: useMenu() only works properly when called inside a component that is a child of Refine, that's why there's a need to use another component as a wrapper of this component.
function SideNavigationContents() {
  const { menuItems } = useMenu();
  const router = useRouter();
  // TODO: uncomment after auth implementation
  // const { mutate: logout } = useLogout();

  return (
    <Layout.Sider
      width={200}
      style={{ height: "100vh", backgroundColor: "var(--bg-white)" }} //
    >
      <div style={{ margin: "24px 24px 12px 24px" }}>
        <HorizontalLogo height={24} />
      </div>

      <Menu
        mode="inline"
        style={{ borderRight: 0 }}
        items={menuItems.map((item) => ({
          key: item.key,
          label: item.label,
          route: item.list,
          icon: item.icon,
        }))}
        // Route navigation
        onClick={({ key }) => {
          const menuItem = menuItems.find((item) => item.key === key);
          if (menuItem?.route) {
            router.push(menuItem.route);
          }
        }}
      ></Menu>

      {/*
      TODO: uncomment after auth implementation
      <button onClick={() => logout()}>Logout</button>
      */}
    </Layout.Sider>
  );
}
