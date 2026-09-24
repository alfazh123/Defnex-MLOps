import { Field, FieldLabel } from "./ui/field";
import { Input } from "./ui/input";

export default function InputSection({
	id,
	label,
	type = "text",
	onChange,
	required = false,
	className,
}: {
	id: string;
	type?: string;
	label: string;
	onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
	required: boolean;
	className?: string;
}) {
	return (
		<Field>
			<FieldLabel>
				{label} {required && <span className="text-red-500">*</span>}
			</FieldLabel>
			<Input
				id={id}
				type={type}
				onChange={onChange}
				required={required}
				className={`${className}`}
			/>
		</Field>
	);
}