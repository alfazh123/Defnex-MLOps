import { createColumnHelper } from "@tanstack/react-table"

import { type DataTableFeatures } from "../table-features"
import type { TrainingRun } from "~/type"
import clsx from "clsx"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "~/components/ui/dropdown-menu";
import { Button } from "~/components/ui/button";
import { EllipsisVertical } from "lucide-react";

const columnHelper = createColumnHelper<DataTableFeatures, TrainingRun>();

export const columnsModelRegistry = (toggleOpen: (id: string) => void) => columnHelper.columns([
    columnHelper.accessor("runId", {
        header: "Run ID",
    }),
    columnHelper.accessor("targetModel", {
        header: "Target Model",
    }),
    columnHelper.accessor("datasetVersion", {
        header: "Dataset Version",
    }),
    columnHelper.accessor("evalLoss", {
        header: "Loss Metric",
        cell: ({ row }) => {
            return (
				<div className="flex items-center">
					{/* <p className="text-sm font-medium text-gray-500">
                    </p> */}
					<span className="text-blue-700 font-bold text-sm">
						{row.original.evalLoss
							? row.original.evalLoss.toFixed(2)
							: "-"}
					</span>
				</div>
			);
        },
    }),
    columnHelper.display({
        id: "actions",
        header: () => (
            <div className="flex justify-center items-center">
                Actions
            </div>
        ),
        cell: ({row}) => {
            return (
				<div className="flex justify-center items-center">
					{/* // ...existing code... */}
					<DropdownMenu>
						<DropdownMenuTrigger
							render={
								<Button
									variant="outline"
									size="icon">
									<EllipsisVertical />
								</Button>
							}
						/>
						<DropdownMenuContent align="end">
							<DropdownMenuItem
								onClick={() => toggleOpen(row.original.runId)}>
								Evaluation
							</DropdownMenuItem>
							<DropdownMenuItem>
								<a
									href={`/promote-deploy/${row.original.runId}`}>
									Promote & Deploy
								</a>
							</DropdownMenuItem>
						</DropdownMenuContent>
					</DropdownMenu>
					{/* // ...existing code... */}
				</div>
			);
        },
    }),
])