#pragma once
#include "../gSLIC_defines.h"
#include "../objects/gSLIC_settings.h"
#include "../objects/gSLIC_spixel_info.h"

namespace gSLIC
{
	namespace engines
	{
		class seg_engine
		{
		protected:

			// normalizing distances
			float max_color_dist;
			float max_xy_dist;

			// images
			UChar4Image *source_img;
			Float4Image *cvt_img;
			IntImage *idx_img;

			// superpixel map
			SpixelMap* spixel_map;
			int spixel_size;

			// real image size
			Vector2i img_size;
			Vector2i map_size;

			objects::settings gslic_settings;

			virtual void Cvt_Img_Space(UChar4Image* inimg, Float4Image* outimg, COLOR_SPACE color_space) = 0;
			virtual void Init_Cluster_Centers() = 0;
			virtual void Find_Center_Association() = 0;
			virtual void Update_Cluster_Center() = 0;
			virtual void Enforce_Connectivity() = 0;

		public:

			seg_engine(const objects::settings& in_settings );
			virtual ~seg_engine();

			const IntImage* Get_Seg_Mask() const {
				idx_img->UpdateHostFromDevice();
				return idx_img;
			};

			void Set_Image_Size(int x, int y) {
				img_size.x = x;
				img_size.y = y;
				map_size.x = (int)ceil((float)x / (float)spixel_size);
				map_size.y = (int)ceil((float)y / (float)spixel_size);
			};

			Vector2i Get_Image_Size() {
				return img_size;
			};

			void Perform_Segmentation(UChar4Image* in_img);
			virtual void Draw_Segmentation_Result(UChar4Image* out_img){};
		};
	}
}

