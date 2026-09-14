import { useState } from "react";
import Layout from "~/components/layout";
import Header from "~/components/ui/header";
import AddDatasetModal from "~/components/modal/add-dataset";
import { datasetFormats, datasets } from "~/utils";
import { DatasetTable } from "~/components/table/dataset-table";
import { columns } from "~/components/table/columns/dataset-column";
import { ScrollArea } from "~/components/ui/scroll-area";

export default function NewKnowledge() {
    const [modalAddKnwledge, setModalAddKnowledge] = useState(false);

    const toggleModalAddKnowledge = () => {
        setModalAddKnowledge((prevState) => !prevState);
        console.log("modalAddKnwledge", modalAddKnwledge);
    }

    return (
		<>
			{/* <div className="w-full h-full flex flex-col overflow-y-auto bg-sidebar rounded-lg"> */}
			<Header
				title="New Knowledge"
				newKnowledge
				modalAddKnowledge={toggleModalAddKnowledge}
			/>
			<ScrollArea>
				<div className="flex-1 max-h-[90vh] min-h-[90vh] h-full p-4">
					<DatasetTable
						columns={columns}
						data={datasets}
					/>
				</div>
			</ScrollArea>

			{/* {modalAddKnwledge && (
                )} */}
			<AddDatasetModal
				isOpen={modalAddKnwledge}
				toggleModal={toggleModalAddKnowledge}
				datasetFormats={datasetFormats}
			/>
			{/* </div> */}
		</>
	);
}