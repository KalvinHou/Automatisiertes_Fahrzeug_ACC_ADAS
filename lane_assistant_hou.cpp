#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <communication/multi_socket.h>
#include <models/tronis/ImageFrame.h>
#include <grabber/opencv_tools.hpp>
#include <models/tronis/BoxData.h>

using namespace std;

class LaneAssistant
{
	// insert your custom functions and algorithms here
	public:

		LaneAssistant()
		{
		}

		double status_ego = 0;
		//@hou 0: before the sign was detected
		//@hou 1: after the sign was detected while the distance is larger than threshold
		//@hou 2: the distance is shorter than threshold yellow white lines exist
		//@hou 3: turn back to 0 (3 is actually 0)

		double status_sign;
		//@hou 0: not detected
		//@hou 1: detected



		//@hou PID parameters
		double preError_steer = 0;
		double integral_steer = 0;
		double preError_speed = 0;
		double integral_speed = 0;
		double preError_speed_distance = 0;
		double integral_speed_distance = 0;


		double percentage_pixel(cv::Mat inputImage)
		{
			double height = inputImage.size().height;
			double width = inputImage.size().width;
			double count = 0;
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					if (inputImage.at<uchar>(i, j) != 0)
					{
						count++;
					}
				}
			}
			double percentage;
			percentage = count / (height * width);
			return percentage;
		}

		// Image Blurring
		cv::Mat deNoise(cv::Mat inputImage)
		{
			cv::Mat output;
			cv::GaussianBlur(inputImage, output, cv::Size(3, 3), 0, 0);

			return output;
		}

		// Edge Detection
		cv::Mat edgeDetector_forAll(cv::Mat image_deNoise)
		{
			cv::Mat output;
			cv::Mat kernel;
			cv::Point anchor;

			// Convert image from RGB to gray
			cv::cvtColor(image_deNoise, output, cv::COLOR_RGB2GRAY);
			// Binarize gray image
			cv::threshold(output, output, 140, 255, cv::THRESH_BINARY);

			// Create a kernel [-1,0,1]
			anchor = cv::Point(-1, -1);
			kernel = cv::Mat(1, 3, CV_32F);
			kernel.at<float>(0, 0) = -1;
			kernel.at<float>(0, 1) = 0;
			kernel.at<float>(0, 2) = 1;

			// Filter the binary image to obtain the edges
			cv::filter2D(output, output, -1, kernel, anchor, 0, cv::BORDER_DEFAULT);

			cv::imshow("All", output);
			
			return output;
		}

		cv::Mat edgeDetector_forWhite(cv::Mat image_deNoise)
		{
			cv::Mat output;
			cv::Mat kernel;
			cv::Point anchor;

			// Convert image from RGB to HSV
			cv::cvtColor(image_deNoise, output, cv::COLOR_BGR2HSV);

			cv::Mat image_white;
			cv::Scalar lower_bound_white = cv::Scalar(0, 0, 150);
			cv::Scalar upper_bound_white = cv::Scalar(180, 30, 255);
			cv::inRange(output, lower_bound_white, upper_bound_white, output);



			// Binarize gray image
			cv::threshold(output, output, 140, 255, cv::THRESH_BINARY);

			// Create a kernel [-1,0,1]
			anchor = cv::Point(-1, -1);
			kernel = cv::Mat(1, 3, CV_32F);
			kernel.at<float>(0, 0) = -1;
			kernel.at<float>(0, 1) = 0;
			kernel.at<float>(0, 2) = 1;

			// Filter the binary image to obtain the edges
			cv::filter2D(output, output, -1, kernel, anchor, 0, cv::BORDER_DEFAULT);

			cv::imshow("White", output);

			return output;
		}

		cv::Mat edgeDetector_forYellow(cv::Mat image_deNoise)
		{
			cv::Mat output;
			cv::Mat kernel;
			cv::Point anchor;

			// Convert image from RGB to HSV
			cv::cvtColor(image_deNoise, output, cv::COLOR_BGR2HSV);

			cv::Mat image_yellow;
			cv::Scalar lower_bound_yellow = cv::Scalar(11, 43, 46);
			cv::Scalar upper_bound_yellow = cv::Scalar(25, 255, 255);
			cv::inRange(output, lower_bound_yellow, upper_bound_yellow, output);



			// Binarize gray image
			cv::threshold(output, output, 140, 255, cv::THRESH_BINARY);

			// Create a kernel [-1,0,1]
			anchor = cv::Point(-1, -1);
			kernel = cv::Mat(1, 3, CV_32F);
			kernel.at<float>(0, 0) = -1;
			kernel.at<float>(0, 1) = 0;
			kernel.at<float>(0, 2) = 1;

			// Filter the binary image to obtain the edges
			cv::filter2D(output, output, -1, kernel, anchor, 0, cv::BORDER_DEFAULT);

			cv::imshow("Yellow", output);

			return output;
		}

		// Mask the Edge Image
		cv::Mat mask(cv::Mat image_edges)
		{
			cv::Mat output;
			cv::Mat mask = cv::Mat::zeros(image_edges.size(), image_edges.type());
			cv::Point pts[8] = { cv::Point(0 * image_edges.size().width, 0.6*image_edges.size().height),
								cv::Point(0 * image_edges.size().width, 0.85*image_edges.size().height),
								cv::Point(0.4*image_edges.size().width, 0.45*image_edges.size().height),
								cv::Point(0.6*image_edges.size().width, 0.45*image_edges.size().height),
								cv::Point(1 * image_edges.size().width, 0.6*image_edges.size().height),
								cv::Point(1 * image_edges.size().width, 0.85*image_edges.size().height),
								cv::Point(0.2*image_edges.size().width, 0.8*image_edges.size().height),
								cv::Point(0.2*image_edges.size().width, 0.8*image_edges.size().height) };
			// Create a binary polygon mask
			cv::fillConvexPoly(mask, pts, 6, cv::Scalar(255, 255, 255));
			// Multiply the edges image and the mask to get the output
			cv::bitwise_and(image_edges, mask, output);



			return output;
		}

		// Hough Lines
		std::vector<cv::Vec4i> houghLines(cv::Mat image_mask)
		{
			std::vector<cv::Vec4i> line;

			HoughLinesP(image_mask, line, 1, CV_PI / 180, 20, 20, 30);
			return line;
		}

		bool left_flag = false;
		bool right_flag = false;

		//Sort Right and Left Lines
		std::vector<std::vector<cv::Vec4i>> lineSeparation(std::vector<cv::Vec4i> lines, cv::Mat image_edges) {
			std::vector<std::vector<cv::Vec4i>> output(2);
			cv::Point initialPoint;
			cv::Point finalPoint;
			double slope_threshold = 0.3;
			std::vector<double> slopes;
			std::vector<cv::Vec4i> selected_lines;				//Vec4i = Vec<int, 4>
			std::vector<cv::Vec4i> right_lines, left_lines;

			for (auto i : lines)
			{
				initialPoint = cv::Point(i[0], i[1]);
				finalPoint = cv::Point(i[2], i[3]);

				double slope = (static_cast<double>(finalPoint.y) - static_cast<double>(initialPoint.y)) /
					(static_cast<double>(finalPoint.x) - static_cast<double>(initialPoint.x) + 0.00001);
				if (std::abs(slope) > slope_threshold)
				{
					slopes.push_back(slope);
					selected_lines.push_back(i);
				}
			}

			int j = 0;
			while (j < selected_lines.size())
			{
				initialPoint = cv::Point(selected_lines[j][0], selected_lines[j][1]);
				finalPoint = cv::Point(selected_lines[j][2], selected_lines[j][3]);

				if (slopes[j] > 0 && finalPoint.x > 0.5*image_edges.size().width && initialPoint.x > 0.55*image_edges.size().width && initialPoint.y > 0.7*image_edges.size().height)
				{
					right_lines.push_back(selected_lines[j]);
					right_flag = true;
				}
				else if (slopes[j] < 0 && finalPoint.x < 0.5*image_edges.size().width && initialPoint.x < 0.45*image_edges.size().width)
				{
					left_lines.push_back(selected_lines[j]);
					left_flag = true;
				}
				j++;
			}
			output[0] = right_lines;
			output[1] = left_lines;
			return output;
		}

		double right_slope, left_slope;
		cv::Point right_point, left_point;
		double right_finalPoint_x, left_finalPoint_x;		// A set for all the x from right_finalPoint


		// Regression for Left and Right Lines
		std::vector<cv::Point> regression(std::vector<std::vector<cv::Vec4i>> right_left_lines, cv::Mat inputImage)
		{
			std::vector<cv::Point> output(4);
			cv::Point left_initialPoint, left_finalPoint, right_initialPoint, right_finalPoint;
			cv::Vec4d right_line, left_line;
			std::vector<cv::Point> right_points, left_points;

			if (right_flag == true)
			{
				for (auto i : right_left_lines[0])
				{
					right_initialPoint = cv::Point(i[0], i[1]);
					right_finalPoint = cv::Point(i[2], i[3]);
					right_points.push_back(right_initialPoint);
					right_points.push_back(right_finalPoint);
				}
				if (right_points.size() > 0)
				{
					cv::fitLine(right_points, right_line, cv::DIST_L2, 0, 0.01, 0.01);
					right_slope = right_line[1] / right_line[0];
					right_point = cv::Point(right_line[2], right_line[3]);
				}
			}

			if (left_flag == true)
			{
				for (auto i : right_left_lines[1])
				{
					left_initialPoint = cv::Point(i[0], i[1]);
					left_finalPoint = cv::Point(i[2], i[3]);
					left_points.push_back(left_initialPoint);
					left_points.push_back(left_finalPoint);
				}
				if (left_points.size() > 0)
				{
					cv::fitLine(left_points, left_line, cv::DIST_L2, 0, 0.01, 0.01);
					left_slope = left_line[1] / left_line[0];
					left_point = cv::Point(left_line[2], left_line[3]);
				}
			}

			int initial_y = inputImage.size().width;
			int final_y = 0.45*inputImage.size().width;

			double right_initialP_x = ((initial_y - right_point.y) / right_slope) + right_point.x;
			double right_finalP_x = ((final_y - right_point.y) / right_slope) + right_point.x;
			double left_initialP_x = ((initial_y - left_point.y) / left_slope) + left_point.x;
			double left_finalP_x = ((final_y - left_point.y) / left_slope) + left_point.x;

			output[0] = cv::Point(right_initialP_x, initial_y);
			output[1] = cv::Point(right_finalP_x, final_y);
			output[2] = cv::Point(left_initialP_x, initial_y);
			output[3] = cv::Point(left_finalP_x, final_y);

			right_finalPoint_x = right_finalP_x;
			left_finalPoint_x = left_finalP_x;

			return output;
		}

		// Plot Result
		void plotLane(cv::Mat inputImage, std::vector<cv::Point> lane)
		{
			std::vector<cv::Point> poly_points;
			cv::Mat output;

			cv::line(inputImage, lane[0], lane[1], cv::Scalar(0, 250, 250), 5, cv::LINE_AA, 0);
			cv::line(inputImage, lane[2], lane[3], cv::Scalar(0, 0, 250), 5, cv::LINE_AA, 0);

			
			inputImage.copyTo(output);
			poly_points.push_back(lane[2]);
			poly_points.push_back(lane[0]);
			poly_points.push_back(lane[1]);
			poly_points.push_back(lane[3]);
			cv::fillConvexPoly(output, poly_points, cv::Scalar(255,250,250), cv::LINE_AA, 0);
			cv::addWeighted(output, 0.3, inputImage, 1.0 - 0.3, 0, inputImage);
			
			//cv::namedWindow("LaneDetection", cv::WINDOW_AUTOSIZE);
			//cv::imshow("LaneDetection", inputImage);
		}



		// ==============================================================================================================================================
		double realTime_distance = 0;


		double Controller(double speed, double distance)
		{
			// @hou two things to do for the throttle
			// @hou maintain a speed
			// @hou maintain a distance towards other vehicle

			// @hou PID parameters for constant speed
			double kp_speed = 0.04;
			double ki_speed = 0.000015;
			double kd_speed = 0.000002;
			double target_speed = 30;
			//double actual_speed = speed;
			double error_speed = 0;

			double out_throttle;


			double kp_speed_distance = 0.01;
			double ki_speed_distance = 0.000001;
			double kd_speed_distance = 0.03;
			double target_distance = 15;
			//double actual_speed = speed;
			double error_speed_distance = 0;


			//if (distance == 0)
			{
				error_speed = target_speed - speed;
				integral_speed += error_speed;

				out_throttle = kp_speed * error_speed + ki_speed * integral_speed + kd_speed * (error_speed - preError_speed);

				preError_speed = error_speed;
			}
			/*
			else
			{
				error_speed_distance = distance - target_distance;
				integral_speed_distance += error_speed_distance;

				out_throttle = kp_speed_distance * error_speed_distance + ki_speed_distance * integral_speed_distance + kd_speed_distance * (error_speed_distance - preError_speed_distance);

				preError_speed_distance = error_speed_distance;
			}
			*/
	
			return out_throttle;
		}


		bool processData(tronis::CircularMultiQueuedSocket& socket)
		{
			// do stuff with data
			// send results via socket
			double outsteer;
			double outsteer_threshold = 1;
			double medium_x;
			
			medium_x = static_cast<double>(right_finalPoint_x + left_finalPoint_x) / 2;

			// @hou PID parameters 
			double kp_steer = 0.002;
			double ki_steer = 0.000015;
			double kd_steer = 0.00002;
			double target_steer = medium_x;
			double current_steer = 0.5*image_.size().width;
			double error_steer = 0;


			error_steer = target_steer - current_steer;
			integral_steer += error_steer;

			outsteer = kp_steer * error_steer + ki_steer * integral_steer + kd_steer * (error_steer - preError_steer);

			preError_steer = error_steer;



			// -----------------------------------------------------------------------------------------------------------------------------------

			cout << "Right now the distance is: "<< realTime_distance << endl;
			double velocity = ego_velocity_*0.036;
			cout << "Right now the velocity is: " << velocity << endl;
			double throttle = Controller(velocity, realTime_distance);

			cout << "THE STATUS OF THE EGO VEHICLE ISSSSSSSSSSSSSSSSSSSSS : " << status_ego << endl;

			socket.send(tronis::SocketData(std::to_string(throttle) + "||" + std::to_string(outsteer)));

			return true;
		}

	protected:
		std::string image_name_;
		cv::Mat image_;
        tronis::LocationSub ego_location_;
        tronis::OrientationSub ego_orientation_;
        double ego_velocity_;

		// Function to detect lanes based on camera image
        // Insert your algorithm here
        void detectLanes()
        {
			// do stuff
			cv::Mat image_deNoise, image_edges, image_mask;
			std::vector<cv::Vec4i> lines;
			std::vector<std::vector<cv::Vec4i>> separatedLines(2);
			std::vector<cv::Point> init_end_points(4);

			double percentage;

			image_deNoise = deNoise(image_);
			if (status_ego == 0)
			{
				image_edges = edgeDetector_forWhite(image_deNoise);
			}
			else if (status_ego == 1)
			{
				image_edges = edgeDetector_forWhite(image_deNoise);
			}
			else if (status_ego == 2)
			{
				image_edges = edgeDetector_forYellow(image_deNoise);
			}

			image_mask = mask(image_edges);
			percentage = percentage_pixel(image_mask);
			cout << "THE PERCENTAGE OF CURRENT FRAME IS: " << percentage << "%" << endl;
			lines = houghLines(image_mask);
			separatedLines = lineSeparation(lines, image_edges);
			init_end_points = regression(separatedLines, image_);
			plotLane(image_, init_end_points);

        }
		
        bool processPoseVelocity( tronis::PoseVelocitySub* msg)
        {
            ego_location_ = msg->Location;
            ego_orientation_ = msg->Orientation;
            ego_velocity_ = msg->Velocity;
            return true;
        }

		bool processBoxData(tronis::BoxDataSub* sensorData)
		{
			for (size_t i = 0; i < sensorData->Objects.size(); i++)
			{
				tronis::ObjectSub& object = sensorData->Objects[i];
				if (object.ActorName.Value().find("Sign") != string::npos)
				{
					status_sign = 1;
					if (status_ego == 0)
					{
						status_ego = 1;
					}
					std::cout << object.ActorName.Value() << " at ";
					std::cout << object.Pose.Location.ToString() << std::endl;
				

					tronis::LocationSub location = object.Pose.Location;

					double pos_x = location.X;
					double pos_y = location.Y;
					realTime_distance = sqrt(pow(location.X, 2) + pow(location.Y, 2))/100;

					if (status_ego == 1 && realTime_distance < 50)
					{
						status_ego = 2;
					}
				}
			}
		}

        bool processObject()
        {
			// do stuff
            return true;
        }

// Helper functions, no changes needed
    public:
		// Function to process received tronis data
		bool getData( tronis::ModelDataWrapper data_model )
		{
            if( data_model->GetModelType() == tronis::ModelType::Tronis )
            {
                std::cout << "Id: " << data_model->GetTypeId()
                            << ", Name: " << data_model->GetName()
                            << ", Time: " << data_model->GetTime() << std::endl;

                // if data is sensor output, process data
                switch( static_cast<tronis::TronisDataType>( data_model->GetDataTypeId() ) )
                {
                    case tronis::TronisDataType::Image:
                    {
                        processImage(
                            data_model->GetName(),
                            data_model.get_typed<tronis::ImageSub>()->Image );
                        break;
                    }
                    case tronis::TronisDataType::ImageFrame:
                    {
                        const tronis::ImageFrame& frames(
                            data_model.get_typed<tronis::ImageFrameSub>()->Images );
                        for( size_t i = 0; i != frames.numImages(); ++i )
                        {
                            std::ostringstream os;
                            os << data_model->GetName() << "_" << i + 1;

                            processImage( os.str(), frames.image( i ) );
                        }
                        break;
                    }
                    case tronis::TronisDataType::ImageFramePose:
                    {
                        const tronis::ImageFrame& frames(
                            data_model.get_typed<tronis::ImageFramePoseSub>()->Images );
                        for( size_t i = 0; i != frames.numImages(); ++i )
                        {
                            std::ostringstream os;
                            os << data_model->GetName() << "_" << i + 1;

                            processImage( os.str(), frames.image( i ) );
                        }
                        break;
                    }
                    case tronis::TronisDataType::PoseVelocity:
                    {
                        processPoseVelocity( data_model.get_typed<tronis::PoseVelocitySub>() );
                        break;
                    }
					case tronis::TronisDataType::BoxData:
					{
						realTime_distance = 0;
						processBoxData(data_model.get_typed<tronis::BoxDataSub>());
					}
                    case tronis::TronisDataType::Object:
                    {
                        processObject();
                        break;
                    }
                    default:
                    {
                        std::cout << data_model->ToString() << std::endl;
                        break;
                    }
                }
                return true;
            }
            else
            {
                std::cout << data_model->ToString() << std::endl;
                return false;
            }
		}

	protected:
		// Function to show an openCV image in a separate window
        void showImage( std::string image_name, cv::Mat image )
        {
            cv::Mat out = image;
            if( image.type() == CV_32F || image.type() == CV_64F )
            {
                cv::normalize( image, out, 0.0, 1.0, cv::NORM_MINMAX, image.type() );
            }
            cv::namedWindow( image_name.c_str(), cv::WINDOW_NORMAL );
            cv::imshow( image_name.c_str(), out );
        }

		// Function to convert tronis image to openCV image
		bool processImage( const std::string& base_name, const tronis::Image& image )
        {
            std::cout << "processImage" << std::endl;
            if( image.empty() )
            {
                std::cout << "empty image" << std::endl;
                return false;
            }

            image_name_ = base_name;
            image_ = tronis::image2Mat( image );

            detectLanes();
            showImage( image_name_, image_ );

            return true;
        }
};

// main loop opens socket and listens for incoming data
int main( int argc, char** argv )
{
    std::cout << "Welcome to lane assistant" << std::endl;

	// specify socket parameters
	std::string socket_type = "TcpSocket";
    std::string socket_ip = "127.0.0.1";
    std::string socket_port = "7778";

    std::ostringstream socket_params;
    socket_params << "{Socket:\"" << socket_type << "\", IpBind:\"" << socket_ip << "\", PortBind:" << socket_port << "}";

    int key_press = 0;	// close app on key press 'q'
    tronis::CircularMultiQueuedSocket msg_grabber;
    uint32_t timeout_ms = 500; // close grabber, if last received msg is older than this param

	LaneAssistant lane_assistant;


	while( key_press != 'q' )
    {
        std::cout << "Wait for connection..." << std::endl;
        msg_grabber.open_str( socket_params.str() );

        if( !msg_grabber.isOpen() )
        {
            printf( "Failed to open grabber, retry...!\n" );
            continue;
        }

        std::cout << "Start grabbing" << std::endl;
		tronis::SocketData received_data;
        uint32_t time_ms = 0;

        while( key_press != 'q' )
        {
			// wait for data, close after timeout_ms without new data
            if( msg_grabber.tryPop( received_data, true ) )
            {
				// data received! reset timer
                time_ms = 0;

				// convert socket data to tronis model data
                tronis::SocketDataStream data_stream( received_data );
                tronis::ModelDataWrapper data_model(
                    tronis::Models::Create( data_stream, tronis::MessageFormat::raw ) );
                if( !data_model.is_valid() )
                {
                    std::cout << "received invalid data, continue..." << std::endl;
                    continue;
                }
				// identify data type
                lane_assistant.getData( data_model );
                lane_assistant.processData( msg_grabber );
            }
            else
            {
				// no data received, update timer
                ++time_ms;
                if( time_ms > timeout_ms )
                {
                    std::cout << "Timeout, no data" << std::endl;
                    msg_grabber.close();
                    break;
                }
                else
                {
                    std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
                    key_press = cv::waitKey( 1 );
                }
            }
        }
        msg_grabber.close();
    }
    return 0;
}
