"use client"

import { useTable, type ColumnDef, type ColumnFiltersState, type RowData } from "@tanstack/react-table"

import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "../ui/table"

import { features, type DataTableFeatures } from "./dataset-table-features"
import { Button } from "../ui/button"
import { Input } from "../ui/input"
import { Fragment, useState } from "react"
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { Calendar } from "../ui/calendar"
import { format } from "date-fns"
import { ScrollArea, ScrollBar } from "../ui/scroll-area"
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "../ui/select"
import {
	CalendarIcon,
	CaretLeftIcon,
	CaretRightIcon,
} from "@phosphor-icons/react/dist/ssr";

interface DatasetTableProps<TData extends RowData> {
    columns: ColumnDef<DataTableFeatures, TData>[]
    data: TData[]
}

export function DatasetTable<TData extends RowData>({
    columns,
    data,
}: DatasetTableProps<TData>) {
    const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
    const [startDate, setStartDate] = useState<Date>();
    const [endDate, setEndDate] = useState<Date>();

    const table = useTable({
        features,
        data,
        columns,
        onColumnFiltersChange: setColumnFilters,
        state: {
            columnFilters,
        }
    })

    // table.setPageSize(10);

    const totalItemsPerPage = [
        { value: 10, label: "10" },
        { value: 15, label: "15" },
        { value: 20, label: "20" },
        { value: 25, label: "25" },
        { value: 30, label: "30" },
    ]

    const handlePageSizeChange = (value: number | null) => {
        console.log("handlePageSizeChange", value);
        if (value !== null && !isNaN(value) && value > 0) {
            table.setPageSize(value);
        }
    }

    return (
		<div className="flex flex-col gap-4 w-full">
			{/* Filter */}
			<div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
				<div className="flex items-center">
					<Input
						placeholder="Filter titles..."
						value={
							(table
								.getColumn("title")
								?.getFilterValue() as string) ?? ""
						}
						onChange={(event) =>
							table
								.getColumn("title")
								?.setFilterValue(event.target.value)
						}
						className="max-w-sm"
					/>
				</div>

				<div className="flex flex-wrap gap-2">
					{/* Start Date */}
					<Popover>
						<PopoverTrigger
							className="rounded-md"
							render={
								<Button
									variant="outline"
									data-empty={!startDate}
									className="justify-start text-left font-normal data-[empty=true]:text-muted-foreground"
								/>
							}>
							<CalendarIcon />
							{startDate ? (
								format(startDate, "PPP")
							) : (
								<span>Start date</span>
							)}
						</PopoverTrigger>
						<PopoverContent className="w-auto p-0">
							<Calendar
								mode="single"
								selected={startDate}
								onSelect={setStartDate}
							/>
						</PopoverContent>
					</Popover>

					{/* End Date */}
					<Popover>
						<PopoverTrigger
							className="rounded-md"
							render={
								<Button
									variant="outline"
									data-empty={!endDate}
									className="justify-start text-left font-normal data-[empty=true]:text-muted-foreground"
								/>
							}>
							<CalendarIcon />
							{endDate ? (
								format(endDate, "PPP")
							) : (
								<span>End date</span>
							)}
						</PopoverTrigger>
						<PopoverContent className="w-auto p-0">
							<Calendar
								mode="single"
								selected={endDate}
								onSelect={setEndDate}
							/>
						</PopoverContent>
					</Popover>
				</div>
			</div>

			{/* Table */}
			<Table className="w-full ">
				<TableHeader>
					{table.getHeaderGroups().map((headerGroup) => (
						<TableRow key={headerGroup.id}>
							{headerGroup.headers.map((header) => {
								return (
									<TableHead key={header.id}>
										{header.isPlaceholder ? null : (
											<table.FlexRender header={header} />
										)}
									</TableHead>
								);
							})}
						</TableRow>
					))}
				</TableHeader>
				<TableBody>
					{table.getRowModel().rows?.length ? (
						table.getRowModel().rows.map((row) => (
							<TableRow
								key={row.id}
								data-state={row.getIsSelected() && "selected"}>
								{row.getVisibleCells().map((cell) => (
									<TableCell key={cell.id}>
										<table.FlexRender cell={cell} />
									</TableCell>
								))}
							</TableRow>
						))
					) : (
						<TableRow>
							<TableCell
								colSpan={columns.length}
								className="h-24 text-center">
								No results.
							</TableCell>
						</TableRow>
					)}
				</TableBody>
			</Table>

			{/* Pagination */}
			<div className="flex sm:items-center justify-between gap-4 py-3 sm:flex-row flex-col">
				<div className="flex flex-1 items-center gap-4 flex-wrap">
					<div className="flex items-center gap-2">
						<Button
							variant="default"
							size="sm"
							onClick={() => table.previousPage()}
							disabled={!table.getCanPreviousPage()}
							className="rounded-md p-0 h-8 w-8">
							<CaretLeftIcon className="h-4 w-4" />
						</Button>
						<Button
							variant="default"
							size="sm"
							onClick={() => table.nextPage()}
							disabled={!table.getCanNextPage()}
							className="rounded-md p-0 h-8 w-8">
							<CaretRightIcon className="h-4 w-4" />
						</Button>
					</div>
					<p className="text-sm font-medium text-muted-foreground">
						Page {table.state.pagination.pageIndex + 1} of{" "}
						{table.getPageCount().toLocaleString()}
					</p>
				</div>
				<div className="flex flex-wrap gap-2 items-center">
					<p className="text-sm text-muted-foreground">Page Size</p>
					<Select
						value={table.state.pagination.pageSize}
						onValueChange={handlePageSizeChange}>
						<SelectTrigger className="w-fit">
							<SelectValue placeholder="Select items per page" />
						</SelectTrigger>
						<SelectContent>
							{totalItemsPerPage.map((item) => (
								<SelectItem
									key={item.value}
									value={item.value}>
									{item.label}
								</SelectItem>
							))}
						</SelectContent>
					</Select>
				</div>
			</div>
		</div>
	);
}