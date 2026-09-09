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

import { features, type DataTableFeatures } from "./table-features";
import { Button } from "../ui/button"
import { Input } from "../ui/input"
import { useState } from "react"
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover"
import { CalendarIcon, ChevronLeft, ChevronRight, Search } from "lucide-react"
import { Calendar } from "../ui/calendar"
import { format } from "date-fns"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select"

interface DatasetTableProps<TData extends RowData> {
    columns: ColumnDef<DataTableFeatures, TData>[]
    data: TData[]
    dataset?: boolean
    trainRun?: boolean
    registry?: boolean
}

export function DataTable<TData extends RowData>({
    columns,
    data,
    dataset,
    trainRun,
    registry,
}: DatasetTableProps<TData>) {
    const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
    const [globalFilter, setGlobalFilter] = useState("")
    const [startDate, setStartDate] = useState<Date>();
    const [endDate, setEndDate] = useState<Date>();

    const table = useTable({
        features,
        data,
        columns,
        onColumnFiltersChange: setColumnFilters,
        state: {
            columnFilters,
            globalFilter,
        },
        globalFilterFn: (row, columnId, filterValue) => {
            const search = String(filterValue).toLowerCase();

            const runId = String(row.getValue("runId") ?? "").toLowerCase();
            const targetModel = String(row.getValue("targetModel") ?? "").toLowerCase();

            return runId.includes(search) || targetModel.includes(search);
        },
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
            <div className={`flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between`}>
                {trainRun && (
                    <>
                        <div className="font-semibold text-sm">
                            <div className="flex flex-col">
                                <div className="font-semibold text-sm">
                                    All training Runs
                                </div>
                                <span className="text-xs text-accent-foreground font-light">
                                    Mock worker auto-progresses PENDING → RUNNING → COMPLETED
                                </span>
                            </div>
                        </div>

                        <div className="flex items-center bg-input/50 rounded-md px-2 py-1 focus:ring-1 focus:ring-ring/50 focus:outline-2 focus:outline-ring">
                            <Search className="h-4 w-4 text-muted-foreground" />
                            <Input
                            placeholder="Filter by ID or Target model"
                            value={globalFilter}
                            onChange={(event) => setGlobalFilter(event.target.value)}
                            className="max-w-sm  bg-transparent active:border-none focus:ring-0 focus:outline-none focus-visible:ring-0 focus:border-none"
                            />
                        </div>
                    </>
                )}

                {registry && (
                    <>
                        <div className="font-semibold text-sm">
                            <div className="flex flex-col">
                                <div className="font-semibold text-sm">
                                    Defnex Support Assistant LLM
                                </div>
                                <span className="text-xs text-accent-foreground font-light">
                                    Specialized 7B Qwen fine-tuned for high-accuracy Indonesian fintech support.
                                </span>
                            </div>
                        </div>

                        <div className="flex items-center bg-input/50 rounded-md px-2 py-1 focus:ring-1 focus:ring-ring/50 focus-within:ring-1 focus-within:ring-ring/50">
                            <Search className="h-4 w-4 text-muted-foreground" />
                            <Input
                            placeholder="Filter by ID or Target model"
                            value={globalFilter}
                            onChange={(event) => setGlobalFilter(event.target.value)}
                            className="max-w-sm bg-transparent active:border-none focus:ring-0 focus-within:ring-0 focus:outline-none focus-visible:ring-0 focus:border-none"
                            />
                        </div>
                    </>
                )}

                {dataset && (
                    <>
                        <div className="flex items-center bg-input/50 rounded-md px-2 py-1 focus:ring-1 focus:ring-ring/50 focus-within:ring-1 focus-within:ring-ring/50">
                            <Search className="h-4 w-4 text-muted-foreground" />
                            <Input
                            placeholder="Filter titles..."
                            value={(table.getColumn("title")?.getFilterValue() as string) ?? ""}
                            onChange={(event) =>
                                table.getColumn("title")?.setFilterValue(event.target.value)
                            }
                            className="max-w-sm  bg-transparent active:border-none focus:ring-0 focus-within:ring-0 focus:outline-none focus-visible:ring-0 focus:border-none"
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
                                    }
                                >
                                    <CalendarIcon />
                                    {startDate ? format(startDate, "PPP") : <span>Start date</span>}
                                </PopoverTrigger>
                                <PopoverContent className="w-auto p-0">
                                    <Calendar mode="single" selected={startDate} onSelect={setStartDate} />
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
                                    }
                                >
                                    <CalendarIcon />
                                    {endDate ? format(endDate, "PPP") : <span>End date</span>}
                                </PopoverTrigger>
                                <PopoverContent className="w-auto p-0">
                                    <Calendar mode="single" selected={endDate} onSelect={setEndDate} />
                                </PopoverContent>
                            </Popover>
                        </div>
                    </>
                )}
            </div>

            {/* Table */}
            <Table className="w-full ">
                <TableHeader className="bg-gray-600">
                {table.getHeaderGroups().map((headerGroup) => (
                    <TableRow key={headerGroup.id}>
                    {headerGroup.headers.map((header) => {
                        return (
                        <TableHead key={header.id} className="text-white">
                            {header.isPlaceholder ? null : (
                            <table.FlexRender header={header} />
                            )}
                        </TableHead>
                        )
                    })}
                    </TableRow>
                ))}
                </TableHeader>
                <TableBody>
                {table.getRowModel().rows?.length ? (
                    table.getRowModel().rows.map((row) => (
                    <TableRow
                        key={row.id}
                        data-state={row.getIsSelected() && "selected"}
                    >
                        {row.getVisibleCells().map((cell) => (
                        <TableCell key={cell.id}>
                            <table.FlexRender cell={cell} />
                        </TableCell>
                        ))}
                    </TableRow>
                    ))
                ) : (
                    <TableRow>
                    <TableCell colSpan={columns.length} className="h-24 text-center">
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
                            className="rounded-md p-0 h-8 w-8"
                        >
                            <ChevronLeft className="h-4 w-4" />
                        </Button>
                        <Button
                            variant="default"
                            size="sm"
                            onClick={() => table.nextPage()}
                            disabled={!table.getCanNextPage()}
                            className="rounded-md p-0 h-8 w-8"
                        >
                            <ChevronRight className="h-4 w-4" />
                        </Button>
                    </div>
                    <p className="text-sm font-medium text-muted-foreground">
                        Page {table.state.pagination.pageIndex + 1} of {table.getPageCount().toLocaleString()}
                    </p>
                </div>
                <div className="flex flex-wrap gap-2 items-center">
                    <p className="text-sm text-muted-foreground">Page Size</p>
                    <Select 
                        value={table.state.pagination.pageSize} 
                        onValueChange={handlePageSizeChange}
                    >
                        <SelectTrigger className="w-fit">
                            <SelectValue placeholder="Select items per page" />
                        </SelectTrigger>
                        <SelectContent>
                            {totalItemsPerPage.map((item) => (
                                <SelectItem key={item.value} value={item.value}>
                                    {item.label}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
            </div>
        </div>
    )
}