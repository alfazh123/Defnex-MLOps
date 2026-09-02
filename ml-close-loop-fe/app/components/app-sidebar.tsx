import {
	Sidebar,
	SidebarContent,
	SidebarFooter,
	SidebarGroup,
	SidebarGroupLabel,
	SidebarHeader,
	SidebarMenu,
	SidebarMenuItemWrap,
	useSidebar,
} from "./ui/sidebar";
import { useLocation } from "react-router"

export function AppSidebar() {

  const location = useLocation();
  const pathname = location.pathname;
  const isActive = (href: string) => pathname == href;

  const mainMenu = [
		{ title: "Dashboard", href: "/" },
		// { title: "Projects", href: "/projects" },
		{ title: "Chat", href: "/chat" },
  ];

  const dataMenu = [
		{ title: "Datasets", href: "/new-knowledge" },
		{ title: "Training Runs", href: "/train-run" },
		{ title: "Model Registry", href: "/model-registry" },
  ];

  const settingsMenu = [
    { title: "Settings", href: "/settings" },
  ]

  useSidebar();

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
							<SidebarMenuItemWrap
								key={item.title}
								isActive={isActive(item.href)}
								title={item.title}
								href={item.href}
							/>
						))}
					</SidebarMenu>
				</SidebarGroup>

				<SidebarGroup>
					<SidebarGroupLabel>Data</SidebarGroupLabel>
					<SidebarMenu>
						{dataMenu.map((item) => (
							<SidebarMenuItemWrap
								key={item.title}
								isActive={isActive(item.href)}
								title={item.title}
								href={item.href}
							/>
						))}
					</SidebarMenu>
				</SidebarGroup>

				<SidebarGroup>
					<SidebarGroupLabel>Settings</SidebarGroupLabel>
					<SidebarMenu>
						{settingsMenu.map((item) => (
							<SidebarMenuItemWrap
								key={item.title}
								isActive={isActive(item.href)}
								title={item.title}
								href={item.href}
							/>
						))}
					</SidebarMenu>
				</SidebarGroup>
			</SidebarContent>
			<SidebarFooter />
		</Sidebar>
  );
}
