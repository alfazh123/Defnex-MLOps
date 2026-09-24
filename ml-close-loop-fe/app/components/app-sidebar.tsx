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
import { useLocation } from "react-router";
import {
	ChartPieSliceIcon,
	ChatsIcon,
	CpuIcon,
	CubeFocusIcon,
	DatabaseIcon,
	GearIcon,
	SidebarIcon, // Import ikon toggle
} from "@phosphor-icons/react/dist/ssr";

export function AppSidebar() {
	const location = useLocation();
	const pathname = location.pathname;
	const isActive = (href: string) => pathname === href;

	// 1. Ambil state dan fungsi toggle dari context
	const { state, toggleSidebar } = useSidebar();
	const isCollapsed = state === "collapsed";

	const mainMenu = [
		{
			title: "Dashboard",
			href: "/",
			icon: <ChartPieSliceIcon size={20} />,
		},
		{ title: "Chat", href: "/chat", icon: <ChatsIcon size={20} /> },
	];

	const dataMenu = [
		{
			title: "Datasets",
			href: "/dataset",
			icon: <DatabaseIcon size={20} />,
		},
		{
			title: "Training Runs",
			href: "/train-run",
			icon: <CpuIcon size={20} />,
		},
		{
			title: "Model Registry",
			href: "/model-registry",
			icon: <CubeFocusIcon size={20} />,
		},
	];

	const settingsMenu = [
		{ title: "Settings", href: "/settings", icon: <GearIcon size={20} /> },
	];

	return (
		<div className="flex h-screen">
			<Sidebar
				collapsible="icon"
				className={`m-4 max-h-[calc(100vh-2rem)] h-full rounded-2xl border-r-0 bg-[#f8f9fa] text-slate-600 flex flex-col overflow-hidden shadow-sm transition-all duration-300 ${
					isCollapsed ? "w-16" : "w-64"
				}`}>
				{/* Header dengan Tombol Toggle (Area Biru) */}
				<SidebarHeader
					className={`${!isCollapsed && "p-4"} flex flex-row items-center justify-between`}>
					{isCollapsed ? (
						<div className="group group-hover:transition-all group-hover:duration-300 flex items-center gap-3">
							<div className="w-8 h-8 group-hover:hidden rounded-full bg-black flex items-center justify-center text-white font-bold text-sm shadow-sm shrink-0 italic">
								D
							</div>
							<button
								onClick={toggleSidebar}
								className="rounded-full group-hover:flex justify-center items-center hidden bg-black w-8 h-8"
								title={
									isCollapsed
										? "Expand Sidebar"
										: "Collapse Sidebar"
								}>
								<SidebarIcon
									size={18}
									className="text-white"
									weight="bold"
								/>
							</button>
						</div>
					) : (
						<>
							<div className="flex items-center gap-3">
								<div className="w-8 h-8 rounded-full bg-black flex items-center justify-center text-white font-bold text-sm shadow-sm shrink-0">
									i
								</div>
								<span className="text-base font-bold tracking-tight text-slate-900 italic whitespace-nowrap">
									Defnex
								</span>
							</div>

							{/* Tombol Toggle yang berpindah ke area biru */}
							<button
								onClick={toggleSidebar}
								className="p-1.5 rounded-lg hover:bg-slate-200/60 text-slate-500 hover:text-slate-900 transition-colors"
								title={
									isCollapsed
										? "Expand Sidebar"
										: "Collapse Sidebar"
								}>
								<SidebarIcon size={18} />
							</button>
						</>
					)}
				</SidebarHeader>

				<SidebarContent className="px-2 space-y-2">
					{/* Group 1 */}
					<SidebarGroup className="p-0">
						{!isCollapsed && (
							<SidebarGroupLabel className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase px-3 mb-1">
								Application
							</SidebarGroupLabel>
						)}
						<SidebarMenu className="space-y-1">
							{mainMenu.map((item) => (
								<SidebarMenuItemWrap
									key={item.title}
									isActive={isActive(item.href)}
									title={item.title}
									href={item.href}
									icon={item.icon}
									isCollapsed={isCollapsed}
								/>
							))}
						</SidebarMenu>
					</SidebarGroup>

					{/* Group 2 */}
					<SidebarGroup className="p-0 pt-2">
						{!isCollapsed && (
							<SidebarGroupLabel className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase px-3 mb-1">
								Data
							</SidebarGroupLabel>
						)}
						<SidebarMenu className="space-y-1">
							{dataMenu.map((item) => (
								<SidebarMenuItemWrap
									key={item.title}
									isActive={isActive(item.href)}
									title={item.title}
									href={item.href}
									icon={item.icon}
									isCollapsed={isCollapsed}
								/>
							))}
						</SidebarMenu>
					</SidebarGroup>

					{/* Group 3 */}
					<SidebarGroup className="p-0 pt-2">
						{!isCollapsed && (
							<SidebarGroupLabel className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase px-3 mb-1">
								Settings
							</SidebarGroupLabel>
						)}
						<SidebarMenu className="space-y-1">
							{settingsMenu.map((item) => (
								<SidebarMenuItemWrap
									key={item.title}
									isActive={isActive(item.href)}
									title={item.title}
									href={item.href}
									icon={item.icon}
									isCollapsed={isCollapsed}
								/>
							))}
						</SidebarMenu>
					</SidebarGroup>
				</SidebarContent>
				<SidebarFooter />
			</Sidebar>
		</div>
	);
}