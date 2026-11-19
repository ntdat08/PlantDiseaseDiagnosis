import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DISEASE_FILES = {
    "apple_diseases.json": {
        "apple_scab": (
            "Apple scab is characterized by olive-green to dark brown "
            "leaf spots with irregular, feathered edges. The spots may "
            "become darker as the disease progresses. Infected apples "
            "develop rough, dark, corky scab lesions on the fruit surface, "
            "which may cause deformation or cracking. These corky fruit "
            "lesions help distinguish apple scab from other apple diseases. "
            "Triệu chứng bệnh ghẻ táo: trên lá xuất hiện các đốm màu xanh "
            "ô liu đến nâu sẫm, mép không đều hoặc có dạng tua. Vết bệnh "
            "có thể chuyển sang màu đen khi phát triển. Trên quả xuất hiện "
            "các vết sần sùi, đóng vảy, màu sẫm, đôi khi khiến quả biến dạng "
            "hoặc nứt. Dấu hiệu đóng vảy trên quả là đặc điểm quan trọng "
            "để nhận biết bệnh ghẻ táo."
        ),
        "apple_black_rot": (
            "Apple black rot causes circular leaf spots known as frog-eye "
            "leaf spots. They have purple or reddish-purple borders and "
            "tan to brown centers. Infected fruits develop brown rot lesions "
            "that may form concentric rings. Fruits can become firm, "
            "shriveled, and mummified while remaining attached to the tree. "
            "These frog-eye leaf spots and mummified fruits are characteristic "
            "signs of black rot. "
            "Triệu chứng bệnh thối đen trên táo: lá xuất hiện các đốm tròn "
            "giống hình mắt ếch, có viền tím hoặc đỏ tím và phần tâm màu "
            "nâu vàng. Quả xuất hiện các vết thối màu nâu, có thể hình thành "
            "những vòng tròn đồng tâm. Quả bị bệnh có thể khô quắt, teo lại "
            "và vẫn treo trên cây. Đốm lá hình mắt ếch là dấu hiệu đặc trưng "
            "giúp phân biệt bệnh thối đen."
        ),
        "apple_cedar_apple_rust": (
            "Cedar-apple rust causes small yellow spots on the upper surface "
            "of apple leaves. These spots enlarge and become bright orange "
            "or yellow-orange, sometimes with a reddish border and small "
            "black dots. Orange tubular or fringed structures may appear "
            "on the underside of the leaves. The orange leaf spots and "
            "orange tube-like structures are characteristic signs of "
            "cedar-apple rust. "
            "Triệu chứng bệnh rỉ sắt tuyết tùng trên táo: mặt trên lá xuất "
            "hiện các đốm nhỏ màu vàng, sau đó phát triển thành màu vàng cam "
            "hoặc cam sáng. Một số vết có viền đỏ và các chấm đen nhỏ. "
            "Mặt dưới lá có thể xuất hiện các cấu trúc dạng ống hoặc tua "
            "màu cam. Các đốm vàng cam kết hợp với cấu trúc dạng ống màu "
            "cam là dấu hiệu đặc trưng của bệnh."
        )
    },
    "tomato_diseases.json": {
        "tomato_early_blight": (
            "Tomato early blight causes dark brown, circular leaf spots "
            "with concentric rings, resembling a target or bullseye. "
            "The lesions usually appear first on older leaves and may "
            "have yellow halos around them. The concentric ring pattern "
            "is the main distinguishing symptom of early blight. "
            "Triệu chứng bệnh đốm vòng sớm trên cà chua: lá xuất hiện "
            "các đốm tròn màu nâu sẫm với nhiều vòng đồng tâm, giống hình "
            "bia bắn. Bệnh thường xuất hiện trước trên các lá già. "
            "Xung quanh vết bệnh đôi khi có quầng vàng. Các vòng tròn "
            "đồng tâm là dấu hiệu đặc trưng để nhận biết bệnh."
        ),
        "tomato_late_blight": (
            "Tomato late blight causes irregular, water-soaked lesions "
            "that expand rapidly across leaf tissue and turn brown or black. "
            "Under cool and humid conditions, white mold-like growth may "
            "appear on the underside of affected leaves. The rapidly "
            "expanding irregular lesions and possible white growth help "
            "distinguish late blight from bacterial spot, which typically "
            "produces numerous small, scattered spots. "
            "Triệu chứng bệnh mốc sương trên cà chua: lá xuất hiện các "
            "vết bệnh không đều, ban đầu úng nước, sau đó lan rộng nhanh "
            "và chuyển màu nâu hoặc đen. Trong điều kiện mát và ẩm, mặt "
            "dưới lá có thể xuất hiện lớp mốc trắng. Vết bệnh lớn, không "
            "đều và lan nhanh là đặc điểm giúp phân biệt bệnh mốc sương "
            "với bệnh đốm vi khuẩn."
        ),
        "tomato_bacterial_spot": (
            "Tomato bacterial spot produces numerous small, dark, "
            "water-soaked spots scattered across the leaves. The spots "
            "may later become brown and dry, sometimes with yellow halos. "
            "Unlike late blight, bacterial spot typically appears as "
            "many separate, small lesions rather than large, rapidly "
            "expanding irregular water-soaked areas. "
            "Triệu chứng bệnh đốm vi khuẩn trên cà chua: lá xuất hiện "
            "nhiều đốm nhỏ màu sẫm, phân bố rải rác. Các đốm ban đầu có "
            "thể úng nước, sau đó chuyển thành màu nâu và khô, đôi khi "
            "có quầng vàng xung quanh. Bệnh thường tạo ra nhiều đốm nhỏ "
            "riêng biệt, khác với các vết bệnh lớn, không đều và lan nhanh "
            "của bệnh mốc sương."
        )
    }
}


def main():
    total_updated = 0

    for filename, updates in DISEASE_FILES.items():
        file_path = BASE_DIR / "knowledge_base" / "documents" / "diseases" / filename

        with open(file_path, "r", encoding="utf-8") as file:
            records = json.load(file)

        updated_ids = set()

        for record in records:
            disease_id = record.get("id")

            if disease_id in updates:
                record["symptoms"] = updates[disease_id]
                updated_ids.add(disease_id)
                total_updated += 1

        missing_ids = set(updates.keys()) - updated_ids

        if missing_ids:
            raise ValueError(
                f"Missing disease IDs in {filename}: {missing_ids}"
            )

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)

        print(f"Updated: {filename}")
        print(f"Records updated: {len(updated_ids)}")

    print("\nTotal disease records updated:", total_updated)
    print("Disease symptom update completed!")


if __name__ == "__main__":
    main()
