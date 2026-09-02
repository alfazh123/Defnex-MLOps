import { AppSidebar } from "./app-sidebar";
import { ScrollArea, ScrollBar } from "./ui/scroll-area";
import { SidebarProvider } from "./ui/sidebar";

export default function Layout({ children }: { children: React.ReactNode }) {
	return (
		<SidebarProvider>
			<AppSidebar />
			<div className="@container/main flex flex-col w-full h-screen p-2">
				<div className="w-full h-full flex flex-col overflow-y-auto bg-sidebar rounded-lg">
					<ScrollArea>
						{children}
						<ScrollBar orientation="vertical" />
					</ScrollArea>
				</div>
			</div>
		</SidebarProvider>
	);
}
