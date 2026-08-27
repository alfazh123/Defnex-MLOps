import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarMenuItemWrap,
} from "./ui/sidebar"
import { useLocation } from "react-router"

export function AppSidebar() {

  const location = useLocation();
  const pathname = location.pathname;
  const isActive = (href: string) => pathname.includes(href);

  const mainMenu = [
    { title: "Dashboard", href: "/" },
    { title: "Projects", href: "/projects" },
    { title: "Chat", href: "/chat" },
  ]

  const dataMenu = [
    { title: "New Knowledge", href: "/new-knowledge" },
    { title: "Model Deployment", href: "/model-deployment" },
  ]

  const settingsMenu = [
    { title: "Settings", href: "/settings" },
  ]

  return (
    <Sidebar>
      <SidebarHeader>
        <div className="flex items-center justify-between">
          <span className="text-lg font-semibold">ML Close loop</span>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Application</SidebarGroupLabel>
          <SidebarMenu>
            {mainMenu.map((item) => (
              // <div className="flex items-center hover:bg-sidebar-accent" key={item.title}>
              //   <div className={`${isActive(item.href) ? "bg-slate-300" : "bg-slate-100"} w-1 h-[80%] rounded-2xl`}></div>
              //   <a href={item.href} className={`block px-3 py-1 text-sm font-medium w-full ${isActive(item.href) ? "text-slate-900" : "text-slate-500"} hover:text-slate-900`}>
              //     <p>
              //       {item.title}
              //     </p>
              //   </a>
              // </div>
              <SidebarMenuItemWrap key={item.title} isActive={isActive(item.href)} title={item.title} href={item.href} />
            ))}
          </SidebarMenu>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Data</SidebarGroupLabel>
          <SidebarMenu>
            {dataMenu.map((item) => (
              // <SidebarMenuItem key={item.title}>
              //   <SidebarMenuButton render={<a href={item.href} />}>
              //     {item.title}
              //   </SidebarMenuButton>
              // </SidebarMenuItem>
              <SidebarMenuItemWrap key={item.title} isActive={isActive(item.href)} title={item.title} href={item.href} />
            ))}
          </SidebarMenu>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Settings</SidebarGroupLabel>
          <SidebarMenu>
            {settingsMenu.map((item) => (
              // <SidebarMenuItem key={item.title}>
              //   <SidebarMenuButton render={<a href={item.href} />}>
              //     {item.title}
              //   </SidebarMenuButton>
              // </SidebarMenuItem>
              <SidebarMenuItemWrap key={item.title} isActive={isActive(item.href)} title={item.title} href={item.href} />
            ))}
          </SidebarMenu>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter />
    </Sidebar>
  )
}
