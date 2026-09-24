import { createColumnHelper } from "@tanstack/react-table";

import { type DataTableFeatures } from "../table-features";
import type { ModelRegistryProp, TrainingRun } from "~/type";
import clsx from "clsx";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";
import { Button } from "~/components/ui/button";
import { DotsThreeCircleVerticalIcon } from "@phosphor-icons/react/dist/ssr";

const columnHelper = createColumnHelper<DataTableFeatures, ModelRegistryProp>();

export const columnsModelRegistry = () =>
	columnHelper.columns([
		columnHelper.accessor("family", {
			header: "Family",
			cell: ({ row }) => {
				return (
					<div className="flex flex-col">
						<p className="font-medium text-gray-900 text-sm">
							{row.original.family.name}
						</p>
						<p className="font-light text-xs">
							{row.original.family.id}
						</p>
					</div>
				);
			},
		}),
		columnHelper.accessor("latestVersion", {
			header: "Latest Version",
			cell: ({ row }) => {
				return (
					<p className="flex flex-col text-xs">
						{row.original.latestVersion}
					</p>
				);
			},
		}),
		columnHelper.accessor("stagingVersion", {
			header: "Staging",
			cell: ({ row }) => {
				return (
					<p className="flex flex-col text-xs">
						{row.original.stagingVersion}
					</p>
				);
			},
		}),
		columnHelper.accessor("productionVersion", {
			header: "Production",
			cell: ({ row }) => {
				return (
					<div className="flex items-center font-mono text-xs">
						{row.original.productionVersion}
					</div>
				);
			},
		}),
		columnHelper.accessor("status", {
			header: "Status",
			cell: ({ row }) => {
				return (
					<div className={`rounded-md font-medium flex`}>
						<span
							className={clsx(
								"px-2 py-1 rounded-md text-[0.8rem] font-medium",
								row.original.status === "candidate" &&
									"bg-slate-100 text-slate-800 border border-slate-300",
								row.original.status === "evaluated" &&
									"bg-blue-100 text-blue-800 border border-blue-300",
								row.original.status === "staging" &&
									"bg-yellow-100 text-yellow-800 border border-yellow-300",
								row.original.status === "production" &&
									"bg-green-100 text-green-800 border border-green-300",
								row.original.status === "archive" &&
									"bg-gray-50 text-gray-500 border border-gray-200",
							)}>
							{row.original.status}
						</span>
					</div>
				);
			},
		}),
		columnHelper.accessor("updatedAt", {
			header: "Updated",
			cell: ({ row }) => {
				const updatedAt = new Date(row.original.updatedAt);
				const formattedDate = updatedAt.toLocaleString("en-US", {
					day: "numeric",
					month: "short",
					year: "numeric",
					// hour: "numeric",
					// minute: "numeric",
					// hour12: true,
				});
				return (
					<div className="flex items-center font-mono text-xs text-accent-foreground">
						{formattedDate}
					</div>
				);
			},
		}),
		columnHelper.display({
			id: "actions",
			header: () => (
				<div className="flex justify-center items-center">Actions</div>
			),
			cell: ({ row }) => {
				return (
					<div className="flex justify-center items-center">
						{/* // ...existing code... */}
						<DropdownMenu>
							<DropdownMenuTrigger
								render={
									<Button
										variant="outline"
										size="icon">
										<DotsThreeCircleVerticalIcon />
									</Button>
								}
							/>
							<DropdownMenuContent align="end">
								<DropdownMenuItem>Evaluation</DropdownMenuItem>
								<DropdownMenuItem>
									Promote & Deploy
								</DropdownMenuItem>
							</DropdownMenuContent>
						</DropdownMenu>
						{/* // ...existing code... */}
					</div>
				);
			},
		}),
	]);
