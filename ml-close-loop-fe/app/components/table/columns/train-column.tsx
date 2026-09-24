import { createColumnHelper } from "@tanstack/react-table";

import { type DataTableFeatures } from "../table-features";
import type { TrainingRun } from "~/type";
import clsx from "clsx";
import { Button } from "~/components/ui/button";
import { EyeIcon } from "@phosphor-icons/react/dist/ssr";

const columnHelper = createColumnHelper<DataTableFeatures, TrainingRun>();

export const columnsTrainRun = (setSelected: (id: string) => void) =>
	columnHelper.columns([
		columnHelper.accessor("runId", {
			header: "Run ID",
		}),
		columnHelper.accessor("targetModel", {
			header: "Target Model",
		}),
		columnHelper.accessor("datasetVersion", {
			header: "Dataset Version",
		}),
		columnHelper.accessor("status", {
			header: "Status",
			cell: ({ row }) => {
				return (
					<div className={`rounded-md font-medium flex`}>
						<span
							className={clsx(
								"px-2 py-1 rounded-md text-[0.8rem] font-medium",
								row.original.status === "COMPLETED" &&
									"bg-green-100 text-green-800",
								row.original.status === "RUNNING" &&
									"bg-blue-100 text-blue-800",
								row.original.status === "PENDING" &&
									"bg-yellow-100 text-yellow-800",
								row.original.status === "FAILED" &&
									"bg-red-100 text-red-800",
							)}>
							{row.original.status}
						</span>
					</div>
				);
			},
		}),
		columnHelper.accessor("trainLoss", {
			header: () => (
				<div className="flex justify-center items-center">
					Loss (Train/Eval)
				</div>
			),
			cell: ({ row }) => {
				return (
					<div className="flex justify-center items-center">
						<p className="text-sm font-medium text-gray-500">
							{row.original.trainLoss
								? row.original.trainLoss.toFixed(2)
								: "-"}
							/
							<span className="text-blue-700 font-bold">
								{row.original.evalLoss
									? row.original.evalLoss.toFixed(2)
									: "-"}
							</span>
						</p>
					</div>
				);
			},
		}),
		columnHelper.accessor("created", {
			header: "Created at",
			cell: ({ row }) => {
				const date = new Date(row.original.created);
				const formattedDate = date.toLocaleString("en-US", {
					// year: "numeric",
					month: "short",
					day: "numeric",
					// hour: "numeric",
					// minute: "numeric",
					// second: "numeric",
				});
				return (
					<div className="flex">
						<span>{formattedDate}</span>
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
						<Button
							variant="ghost"
							onClick={() => setSelected(row.original.runId)}
							className="font-medium text-sm h-8 w-8">
							<EyeIcon className="h-4 w-4" />
						</Button>
					</div>
				);
			},
		}),
	]);
