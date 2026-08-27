import { AppSidebar } from "./app-sidebar";
import MainContent from "./main-content";
import { SidebarProvider, SidebarTrigger } from "./ui/sidebar";

export default function Layout({ children, title }: { children: React.ReactNode, title: string }) {
  return (
    <SidebarProvider>
      <AppSidebar />
      <div className="flex flex-col w-full h-screen p-2">
        {/* <SidebarTrigger />
        {children} */}
        <MainContent title={title}>
          {children}
        </MainContent>
      </div>
    </SidebarProvider>
  );
}
